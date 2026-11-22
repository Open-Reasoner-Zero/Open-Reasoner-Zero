import asyncio
import logging
import os
import socket
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_ulong, c_void_p
from typing import Dict, Optional, Type, Union

import deepspeed
import ray
import torch
import torch.distributed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from ray.util.placement_group import PlacementGroup, PlacementGroupSchedulingStrategy, placement_group
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

# from openrlhf.models import Actor
from transformers.trainer import get_scheduler

from orz.exp_engine.parallels.orz_distributed_c10d import CUDAIPCHandle, orz_init_process_group
from orz.ppo.models import Actor, get_llm_for_sequence_regression
from orz.ppo.replay_buffer import Experience
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from orz.ppo.utils import masked_mean

_SET_AFFINITY = False


# Adapt from OpenRLHF
class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


# Adapt from OpenRLHF
class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


# Adapt from OpenRLHF
class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


# Adapt from OpenRLHF
class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_ratio = log_probs - old_log_probs
        if action_mask is not None:
            # Excluded positions can contain padding sentinels or extreme logits.
            log_ratio = torch.where(action_mask.bool(), log_ratio, torch.zeros_like(log_ratio))
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


class RayActor(BasePPORole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def _set_numa_affinity(self, rank):
        def local_rank_to_real_gpu_id(local_rank):
            cuda_visible_devices = [
                int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
            ]
            return cuda_visible_devices[local_rank]

        rank = local_rank_to_real_gpu_id(rank)

        global _SET_AFFINITY
        if _SET_AFFINITY:
            return

        from ctypes.util import find_library

        class bitmask_t(Structure):
            _fields_ = [
                ("size", c_ulong),
                ("maskp", POINTER(c_ulong)),
            ]

        LIBNUMA = CDLL(find_library("numa"))
        LIBNUMA.numa_parse_nodestring.argtypes = [c_char_p]
        LIBNUMA.numa_parse_nodestring.restype = POINTER(bitmask_t)
        LIBNUMA.numa_run_on_node_mask.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_run_on_node_mask.restype = c_int
        LIBNUMA.numa_set_membind.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = c_void_p
        LIBNUMA.numa_num_configured_nodes.argtypes = []
        LIBNUMA.numa_num_configured_nodes.restype = c_int

        def numa_bind(nid: int):
            bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
            LIBNUMA.numa_run_on_node_mask(bitmask)
            LIBNUMA.numa_set_membind(bitmask)

        numa_nodes = LIBNUMA.numa_num_configured_nodes()
        num_gpu_pre_numa_node = 8 // numa_nodes
        numa_bind(self._local_rank // num_gpu_pre_numa_node)
        _SET_AFFINITY = True

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        """This function guaratees the memory are all released (only torch context cache <100M will remain)."""
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        if isinstance(self.model, Actor):
            model = self.model.model
        else:
            model = self.model

        if model.zero_optimization_stage() == 3:
            from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

            model.optimizer.offload_states(
                include=[
                    OffloadStateTypeEnum.optim_states,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                    OffloadStateTypeEnum.hp_params,
                    # OffloadStateTypeEnum.lp_grads,
                    # OffloadStateTypeEnum.lp_params, # dangerous
                ],
                device=OffloadDeviceEnum.cpu,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage 2 is not supported yet")

    def backload_to_gpu(self, non_blocking=True):
        # NOTE: this function reloads the weights, ensuring the calculation
        if isinstance(self.model, Actor):
            model = self.model.model
        else:
            model = self.model
        if model.zero_optimization_stage() == 3:
            model.reload_states(non_blocking=non_blocking)
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage 2 is not supported yet")


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BasePPORole],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    async def offload_to_cpu(self):
        await asyncio.gather(*[actor.offload_to_cpu.remote() for actor in self._actor_handlers])

    async def backload_to_gpu(self):
        await asyncio.gather(*[actor.backload_to_gpu.remote() for actor in self._actor_handlers])

    async def async_save_model(self, tokenizer, iteration):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        save_tasks = [actor.save_model.remote(tokenizer, iteration) for actor in self._actor_handlers]
        return await asyncio.gather(*save_tasks)

    async def async_ppo_train(self, global_steps, replay_buffers):
        return await asyncio.gather(
            *[actor.ppo_train.remote(global_steps, replay_buffers[i]) for i, actor in enumerate(self._actor_handlers)]
        )

    async def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return await asyncio.gather(*refs)


class PolicyRayActorBase(RayActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self.args = strategy.args
        self._setup_distributed(strategy)

        ds_config = strategy.get_ds_train_config(is_actor=True)
        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            target_modules=strategy.args.target_modules,
            ds_config=ds_config,
            packing_samples=True,
        )

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=self.args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=self.args.l2
        )

        actor_scheduler = get_scheduler(
            "constant_with_warmup", actor_optim, num_warmup_steps=self.args.num_warmup_steps
        )

        if self.args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(self.args.ckpt_path, "_actor")
        if self.args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.model.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            self.strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # set ppo loss function
        self.actor_loss_fn = PolicyLoss(self.args.eps_clip)

    def save_model(self, tokenizer, iteration):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.model,
            tokenizer,
            os.path.join(args.save_path, f"iter{iteration}", "policy"),
        )

    def forward(
        self, sequences, num_actions, attention_mask, return_output=False, ring_attn_group=None, packed_seq_lens=None
    ):
        device = torch.cuda.current_device()
        self.model.eval()
        with torch.no_grad():
            policy_logprob = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output,
                ring_attn_group,
                packed_seq_lens,
            )
        return policy_logprob.to("cpu")

    def ppo_train(self, global_steps, replay_buffer):
        # replay buffer may be empty at first, we should rebuild at each training
        device = torch.cuda.current_device()
        dataloader = DataLoader(
            replay_buffer,
            batch_size=replay_buffer.sample_batch_size,
            drop_last=False,
            collate_fn=replay_buffer.collate_fn,
            pin_memory=False,
        )

        update_steps = self.args.policy_update_steps
        accumulation_steps = max(1, len(dataloader) // update_steps)

        status_list = []
        status_mean = {}
        policy_update_steps = 0
        for epoch in range(self.args.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Actor Train epoch [{epoch + 1}/{self.args.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, global_steps, local_step, accumulation_steps)
                policy_update_steps += 1

                # for DP
                status = self.strategy.all_reduce(status)

                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                        "ent": status["entropy"],
                    }
                    if "reward" in status:
                        short_status["rm"] = status["reward"]
                    if "avg_custom_rewards" in status:
                        short_status["avg_custom_rewards"] = status["avg_custom_rewards"]

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)
                if (local_step + 1) // accumulation_steps == update_steps:
                    break

        torch.distributed.barrier()

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        status_mean["policy_update_steps"] = policy_update_steps / accumulation_steps
        return status_mean

    def training_step(self, experience: Experience, global_steps, local_step, accumulation_steps) -> Dict[str, float]:
        self.model.train()

        # TODO: only support packed sequences for now
        assert isinstance(experience.sequences, list)
        sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
        old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
        base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
        num_actions = torch.cat(experience.num_actions, dim=0).long().tolist()
        packed_seq_lens = torch.cat(experience.packed_seq_lens, dim=0).long().tolist()
        attention_mask = torch.cat(experience.attention_mask, dim=0).unsqueeze(0)

        # actor loss
        action_log_probs, output = self.model(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        # TODO: recompute advantages
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )
        # clip ratio
        with torch.no_grad():
            ratio = (action_log_probs - old_action_log_probs).exp()
            clamp_ratio = ratio.clamp(1 - self.args.eps_clip, 1 + self.args.eps_clip)
            clip_ratio = (clamp_ratio != ratio).sum().item() / ratio.numel()

        # entropy
        with torch.no_grad():
            assert isinstance(experience.sequences, list), "Only support packed sequences"
            action_logits = output["logits"][:, :-1, :]
            action_log_probs_all = torch.nn.functional.log_softmax(action_logits, dim=-1)

            action_log_probs_all_list = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs_all_list.append(action_log_probs_all[:, start:end])
                offset += seq_len
            action_log_probs_all = torch.cat(action_log_probs_all_list, dim=1)

            # Calculate entropy in chunks to avoid OOM
            chunk_size = 512  # Adjust this value based on your GPU memory
            num_chunks = (action_log_probs_all.size(1) + chunk_size - 1) // chunk_size
            entropy_sum = 0
            total_tokens = 0

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, action_log_probs_all.size(1))
                chunk = action_log_probs_all[:, start_idx:end_idx]

                # Calculate entropy for this chunk
                chunk_probs = chunk.exp()
                chunk_entropy = -(chunk_probs * chunk).sum(-1)
                entropy_sum += chunk_entropy.sum().item()
                total_tokens += chunk_entropy.numel()

            entropy = entropy_sum / total_tokens

        # kl loss
        if self.args.use_kl_loss:
            kl_loss = action_log_probs - base_action_log_probs
            if self.args.use_kl_estimator_k3:
                kl_loss = -kl_loss
                r = kl_loss.exp()
                kl_loss = r - 1.0 - kl_loss
            kl_loss = masked_mean(kl_loss, experience.action_mask, dim=-1).mean()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * self.args.kl_loss_coef
        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        if (local_step + 1) % accumulation_steps == 0:
            self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        # status
        status = {
            "policy_loss": actor_loss.item(),
            "actor_lr": self.scheduler.get_last_lr()[0],
            "clip_ratio": clip_ratio,
            "entropy": entropy,
        }

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)

    def _set_pad_token_id(self, pad_token_id):
        self.model.model.config["pad_token_id"] = pad_token_id

    def _init_vllm_engines_actor_group(self, vllm_engines=None):
        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines

        if vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            import vllm

            if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
                backend = "gloo"
                self.strategy.print(
                    "WARNING:using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
                )

            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            self._model_update_group = orz_init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )

            ray.get(refs)
        torch.distributed.barrier()

    def _broadcast_to_vllm(self, vllm_engines):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in vllm_engines
                ]
            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)
        self.strategy.print("Broadcast actor weights to vllm engines done")

    def _broadcast_to_vllm_cudaipc(self, vllm_engines):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                rank = torch.distributed.get_rank()
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    vllm_engines[rank].update_weight_internal_with_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        cudaipc_handler=CUDAIPCHandle.from_tensor(param.data),
                        empty_cache=count == num_params,
                    )
                ]
                ray.get(refs)

        self.strategy.print("Broadcast actor weights to vllm engines done")

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model = self.model.model.module
        for name, param in model.named_parameters():
            # 计算关键统计信息
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                tensor_stats = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "norm": param.data.norm().item(),
                    "shape": tuple(param.shape),
                    # 可选：计算一些极值
                    "max": param.data.max().item(),
                    "min": param.data.min().item(),
                }
                stats[name] = tensor_stats

        return stats


class CriticRayActorBase(RayActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args
        self.args = args

        self._setup_distributed(strategy)

        ds_config = strategy.get_ds_train_config(is_actor=False)
        with torch.device("meta"):
            AutoModel.from_pretrained(pretrain, trust_remote_code=True)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            target_modules=strategy.args.target_modules,
            ds_config=ds_config,
            value_head_prefix=strategy.args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
            packing_samples=True,
        )
        # configure optimizer
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            "constant_with_warmup",
            critic_optim,
            num_warmup_steps=self.args.num_warmup_steps,
        )

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.load_ckpt(self.model, ckpt_path)
            self.strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # set ppo loss function
        self.critic_loss_fn = ValueLoss(args.value_clip)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.model.eval()
        with torch.no_grad():
            value = self.model(
                sequences.to(device), num_actions, attention_mask.to(device), packed_seq_lens=packed_seq_lens
            )
        self.model.train()  # reset model state
        return value.to("cpu")

    def save_model(self, tokenizer, iteration):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.model,
            tokenizer,
            os.path.join(args.save_path, f"iter{iteration}", "critic"),
        )

    def ppo_train(self, global_steps, replay_buffer):
        torch.cuda.empty_cache()
        self.model.train()

        dataloader = DataLoader(
            replay_buffer,
            batch_size=replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            collate_fn=replay_buffer.collate_fn,
        )

        device = torch.cuda.current_device()
        update_steps = self.args.critic_update_steps
        accumulation_steps = max(1, len(dataloader) // update_steps)

        status_list = []
        status_mean = {}
        critic_update_steps = 0
        for epoch in range(self.args.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Critic Train epoch [{epoch + 1}/{self.args.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, global_steps, local_step, accumulation_steps)
                critic_update_steps += 1

                # for DP
                status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)

                if (local_step + 1) // accumulation_steps == update_steps:
                    break

        torch.distributed.barrier()
        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)

        status_mean["critic_update_steps"] = critic_update_steps / accumulation_steps
        return status_mean

    def training_step(self, experience: Experience, global_steps, local_step, accumulation_steps) -> Dict[str, float]:

        assert isinstance(experience.sequences, list)
        sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
        old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
        returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
        num_actions = torch.cat(experience.num_actions, dim=0).long().tolist()
        packed_seq_lens = torch.cat(experience.packed_seq_lens, dim=0).long().tolist()
        attention_mask = torch.cat(experience.attention_mask, dim=0).unsqueeze(0)

        # critic loss
        values, output = self.model(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )

        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)
        if (local_step + 1) % accumulation_steps == 0:
            self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="critic")

        # status
        status = {
            "critic_loss": loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.scheduler.get_last_lr()[0],
        }
        return status


class RewardRayActorBase(RayActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        with torch.device("meta"):
            AutoModel.from_pretrained(pretrain, trust_remote_code=True)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=True,
        )
        if strategy.args.ref_reward_offload or strategy.args.colocate_all:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        num_actions=None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device), packed_seq_lens=packed_seq_lens)
        return reward.to("cpu")


class RefRayActorBase(RayActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=True,
        )

        if strategy.args.ref_reward_offload or strategy.args.colocate_all:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")


PolicyRayActor = ray.remote(num_gpus=1)(PolicyRayActorBase)
CriticRayActor = ray.remote(num_gpus=1)(CriticRayActorBase)
RewardRayActor = ray.remote(num_gpus=1)(RewardRayActorBase)
RefRayActor = ray.remote(num_gpus=1)(RefRayActorBase)
