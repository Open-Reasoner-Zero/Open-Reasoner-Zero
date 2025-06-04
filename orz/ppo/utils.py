import re
import time
from collections import defaultdict
from typing import Optional, Tuple, Union

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.nn.utils.rnn import pad_sequence

from orz.exp_engine.accelerators.inference.vllm_engine import LLMActor
from orz.ppo.openrlhf_deepspeed import DeepspeedStrategy

LLMRayActor = ray.remote(LLMActor)


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


class Timer:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"{self.message}, time cost: {time.time() - self.start_time:.2f}s")


def _validate_args(args: DictConfig):
    assert args.zero_stage != 3 or args.vllm_num_engines > 0, "ZeRO-3 is only supported when vLLM enabled"
    assert not (
        args.reward_pretrain is None and not args.use_compute_reward_fn
    ), "at least one of reward model or custom reward fn should be set."

    assert (
        args.packing_max_len >= args.prompt_max_len + args.generate_max_len
    ), "packing_max_len should be set greater than prompt_max_len + generate_max_len when packing samples is True"
    assert (
        args.micro_forward_batch_size == 1 and args.micro_train_batch_size == 1
    ), "micro_forward_batch_size and micro_train_batch_size should be 1 when packing samples is True"


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
    use_abs_kl: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    if use_abs_kl:
        log_ratio = log_ratio.abs()

    return log_ratio


@ray.remote(num_cpus=1)
@torch.no_grad()
def compute_reward(
    r: Optional[Union[torch.Tensor, float]],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    custom_rewards: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
    use_kl_loss: bool = False,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if r is not None and reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        if not use_kl_loss:
            kl_reward = -kl_coef * kl
        else:
            kl_reward = torch.zeros_like(kl)
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        if r is not None:
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
            reward = last_reward + kl_reward
        else:
            reward = kl_reward
        if custom_rewards is not None:
            custom_rewards_batch = pad_sequence(custom_rewards, batch_first=True, padding_value=0.0)
            reward = reward + custom_rewards_batch
    else:
        if not use_kl_loss:
            kl_reward = -kl_coef * kl
        else:
            kl_reward = torch.zeros_like(kl)
        if r is not None:
            kl_reward[:, torch.tensor(num_actions).cumsum(dim=-1) - 1] += r

        if custom_rewards is not None:
            custom_rewards = torch.cat(custom_rewards, dim=0)
            reward = kl_reward + custom_rewards.unsqueeze(0)
        else:
            reward = kl_reward

    return reward


@ray.remote(num_cpus=1)
@torch.no_grad()
def get_advantages_and_returns(
    values: Optional[torch.Tensor],
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    num_actions: Optional[torch.Tensor],
    gamma: float,
    lambd: float,
    packing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (batch_size, response_size)
    - rewards: Tensor of shape (batch_size, response_size)

    Output:
    - advantages: Tensor of shape (batch_size, response_size)
    - returns: Tensor of shape (batch_size, response_size)
    """

    # fp32 mode
    if values is not None:
        values = values.float()
    rewards = rewards.float()

    if packing:
        accum_reverse_num_actions = torch.cumsum(torch.tensor(num_actions), dim=0)
        sample_idx = len(num_actions) - 1
    else:
        accum_reverse_num_actions = None

    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    # Mask invalid responses
    if action_mask is not None:
        if values is not None:
            values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        if values is not None:
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        else:
            nextvalues = 0.0
        if packing and sample_idx >= 0 and t + 1 == accum_reverse_num_actions[sample_idx]:
            sample_idx -= 1
            lastgaelam = 0
            nextvalues = 0.0
        if values is not None:
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        else:
            delta = rewards[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    if values is not None:
        returns = advantages + values
    else:
        returns = advantages
    return advantages.detach(), returns


def normalize_advantages(buffer):
    items = []
    action_masks = []
    for item in buffer:
        items.append(getattr(item, "advantages"))
        action_masks.append(item.action_mask)

    items_vector = torch.cat(items).float().flatten()

    if action_masks[0] is None:
        # packing samples has no action mask
        action_masks_vector = 1
        num_actions = items_vector.numel()
    else:
        action_masks_vector = torch.cat(action_masks).flatten()
        num_actions = action_masks_vector.sum()

    # mean
    mean = items_vector.mean()
    # std
    std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
    rstd = (std / num_actions).clamp(min=1e-8).rsqrt()

    for i, item in enumerate(buffer):
        t = (items[i] - mean) * rstd
        setattr(item, "advantages", t.bfloat16())
    return buffer


class ORZDeepspeedStrategy(DeepspeedStrategy):
    def get_ds_train_config(self, is_actor):
        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config


def get_strategy(args):
    strategy = ORZDeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    colocate_with_actor: bool,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    colocate_pg: Optional[PlacementGroup] = None,
    vllm_impl: str = "old"
):
    if vllm_impl == "colocate":
        return _create_colocate_vllm_engines(
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            pretrain=pretrain,
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            colocate_pg=colocate_pg,
        )
    vllm_engines = []
    if tensor_parallel_size > 1:
        assert not colocate_with_actor, "colocate_with_actor is not supported when tensor_parallel_size > 1"
        num_gpus = 0
        for i in range(num_engines):
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
            vllm_engines.append(
                LLMRayActor.options(num_cpus=1, num_gpus=num_gpus, scheduling_strategy=scheduling_strategy,).remote(
                    pretrain,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype="bfloat16",
                    seed=seed + i,
                    enable_prefix_caching=enable_prefix_caching,
                    enforce_eager=enforce_eager,
                    max_model_len=max_model_len,
                    enable_chunked_prefill=enable_chunked_prefill,
                    max_num_batched_tokens=max_num_batched_tokens if enable_chunked_prefill else None,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                    block_size=256,
                )
            )
    else:
        if not colocate_with_actor:
            num_gpus = 1
            num_cpus = 1
            bundles = [{"GPU": 1, "CPU": 1}] * num_engines
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        else:
            num_gpus = 0.2
            num_cpus = 0.2
            assert colocate_pg is not None, "colocate_pg must be provided when colocate_with_actor is True"

        for i in range(num_engines):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=colocate_pg if colocate_with_actor else pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=i,
            )
            vllm_engines.append(
                LLMRayActor.options(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    pretrain,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype="bfloat16",
                    seed=seed + i,
                    enable_prefix_caching=enable_prefix_caching,
                    enforce_eager=enforce_eager,
                    max_model_len=max_model_len,
                    enable_chunked_prefill=enable_chunked_prefill,
                    max_num_batched_tokens=max_num_batched_tokens if enable_chunked_prefill else None,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                    block_size=256,
                )
            )
        if colocate_with_actor:
            offload_refs = []
            for llm in vllm_engines:
                offload_refs.append(llm.offload_to_cpu.remote())
            ray.get(offload_refs)
            logger.info("Offloaded all vLLM engines to CPU")

    return vllm_engines


# reflection pattern checking related

# check how many reflection pattern related words are in the responses
def check_reflection_pattern(response: str) -> dict[str, int]:
    # TODO: may need to add more pattern
    reflection_pattern_words = [
        r"wait,",
        r"recheck[,\s]",
        r"retry",
        r"alternatively,",
        r"however,",
    ]
    res = defaultdict(int)
    for word in reflection_pattern_words:
        # can only be followed by a comma or a space
        res[word] = len(re.findall(word, response))
    return res


def _create_colocate_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    colocate_pg: Optional[PlacementGroup] = None,
):
    from orz.exp_engine.accelerators.inference.vllm_engine import ColocateVllmActor

    vllm_engines = []

    bundles_to_node_id = ray.util.placement_group_table(colocate_pg)["bundles_to_node_id"]
    nodes = ray.nodes()

    port = 51995  # TODO (zhanghan): make it adaptive

    for i in range(num_engines):
        offset = i * tensor_parallel_size

        actors = []
        for local_rank in range(tensor_parallel_size):
            node_id = bundles_to_node_id[offset]
            for node in nodes:
                if node["NodeID"] == node_id:
                    ip = node["NodeManagerAddress"]
                    break
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=colocate_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=offset + local_rank,
            )
            actor = (
                ray.remote(ColocateVllmActor)
                .options(
                    num_cpus=0.1,
                    num_gpus=0.1,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env={
                        "env_vars": {
                            "RANK": str(local_rank),
                            "LOCAL_RANK": str(local_rank),
                            "WORLD_SIZE": str(tensor_parallel_size),
                            "MASTER_ADDR": str(ip),
                            "MASTER_PORT": str(port + i),
                        }
                    },
                )
                .remote(
                    model_path=pretrain,
                    tensor_parallel_size=tensor_parallel_size,
                    seed=seed + i,
                    enable_prefix_caching=enable_prefix_caching,
                    enforce_eager=True,
                    max_model_len=max_model_len,
                    enable_chunked_prefill=enable_chunked_prefill,
                    max_num_batched_tokens=max_num_batched_tokens if enable_chunked_prefill else None,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                    block_size=256,
                )
            )

            actors.append(actor)

        vllm_engines.append(actors)

    # Offload
    offload_refs = []
    for engines in vllm_engines:
        for tp_engine in engines:
            offload_refs.append(tp_engine.offload_to_cpu.remote())
    ray.get(offload_refs)

    return vllm_engines


def _get_master_addr_port_by_bundle_helper():
    import socket

    ip = socket.gethostbyname(socket.gethostname())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return ip, port


def get_master_addr_port_by_bundle(pg, bundle_index):
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=bundle_index
    )
    return ray.get(
        ray.remote(_get_master_addr_port_by_bundle_helper)
        .options(
            num_cpus=0.01,
            scheduling_strategy=scheduling_strategy,
        )
        .remote()
    )


def detokenize_vllm_outputs(responses, tokenizer):
    out_token_ids = []
    for resp in responses:
        for output in resp.outputs:
            out_token_ids.append(output.token_ids)
    out_texts = tokenizer.batch_decode(out_token_ids)
    i = 0
    for resp in responses:
        for output in resp.outputs:
            output.text = out_texts[i]
            i += 1
    return responses


def process_stop(responses, stop, tokenizer):
    out_texts = []
    for resp in responses:
        for output in resp.outputs:
            for s in stop:
                if s in output.text:
                    output.text = output.text.split(s)[0] + s
                    output.stop_reason = s
                    output.finish_reason = "stop"
            out_texts.append(output.text)
    token_ids = tokenizer(out_texts).input_ids
    i = 0
    for resp in responses:
        for output in resp.outputs:
            output.token_ids = token_ids[i]
            i += 1
    return responses
