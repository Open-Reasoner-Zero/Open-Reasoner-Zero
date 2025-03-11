"""
Qwen2.5-7B base model + ppo implemented with a new vllm engine and hsdp

debug running command in single node:
DEBUG_MODE=True python -m playground.orz_7b_colocate_hsdp

Multi-node Training:

on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`
then on master node, run `python -m playground.orz_7b_colocate_hsdp`

"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import ray
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf.listconfig import ListConfig
from openrlhf.models import Actor
from openrlhf.models import PolicyLoss
from ray.util.placement_group import PlacementGroup, placement_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import override

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOLlamaExpConfig
from orz.ppo.actors import CriticRayActorBase, PolicyRayActorBase, PPORayActorGroup
from orz.ppo.utils import create_vllm_engines, get_master_addr_port_by_bundle
from playground.orz_7b_ppo import CustomRewardTrainer, CustomDataset, EvalCustomDataset

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() in ["true", "1"]

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOLlamaExpConfig(BasePPOLlamaExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    train_num_nodes: int = 32 if not DEBUG_MODE else 16
    # resource related settings
    ref_num_nodes: int = train_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = train_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = train_num_nodes
    critic_num_gpus_per_node: int = 1
    reward_num_nodes: int = train_num_nodes
    reward_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = 16 if not DEBUG_MODE else 8
    vllm_tensor_parallel_size: int = 2
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B" # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"orz_ckpt/{file_name}"
    save_path: str = f"orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "data/orz_math_57k_collected.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 20
    rollout_batch_size: int = 128 if not DEBUG_MODE else 8
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 32
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 32

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = not DEBUG_MODE
    eval_interval: int = 10 if not DEBUG_MODE else 5

    # generate related settings
    generate_max_len: int = 8000 if not DEBUG_MODE else 512
    max_len: int = 8192  # TODO: change to larger later
    packing_samples: bool = True
    packing_max_len: int = generate_max_len + prompt_max_len
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = False

    gpu_memory_utilization: float = 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0

    vllm_impl: str = "colocate"
    num_total_gpus: int = 32 if not DEBUG_MODE else 16
    num_ref_shards: int = 8
    num_critic_shards: int = 8
    num_actor_shards: int = 8


class BaseFsdpRayActor:
    def __init__(self):
        # We index the os.environ to ensure the variables are set
        self.global_rank = int(os.environ["GLOBAL_RANK"])
        self.global_world_size = int(os.environ["GLOBAL_WORLD_SIZE"])
        self.shard_master_addr = os.environ["SHARD_MASTER_ADDR"]
        self.shard_port = int(os.environ["SHARD_PORT"])
        self.shard_rank = int(os.environ["SHARD_RANK"])
        self.shard_world_size = int(os.environ["SHARD_WORLD_SIZE"])

        self.replicate_master_addr = os.getenv("REPLICATE_MASTER_ADDR")
        self.replicate_port = os.getenv("REPLICATE_PORT")
        self.replicate_rank = os.getenv("REPLICATE_RANK")
        self.replicate_world_size = os.getenv("REPLICATE_WORLD_SIZE")
        if self.replicate_port:
            self.replicate_port = int(self.replicate_port)
            self.replicate_rank = int(self.replicate_rank)
            self.replicate_world_size = int(self.replicate_world_size)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def offload_to_cpu(self):
        # Offload model parameters to CPU
        for _, param in self.model.named_parameters():
            if hasattr(param, "_local_shard"):
                param._local_shard = param._local_shard.to("cpu", non_blocking=True)
            param.data = param.data.to("cpu", non_blocking=True)
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        for _, buffer in self.model.named_buffers():
            buffer.data = buffer.data.to("cpu", non_blocking=True)
        # Offload optimizer states to CPU
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    state = self.optimizer.state[param]
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.data = value.data.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

    def backload_to_gpu(self):
        # Backload model parameters to GPU
        for _, param in self.model.named_parameters():
            if hasattr(param, "_local_shard"):
                param._local_shard = param._local_shard.to("cuda", non_blocking=True)
            param.data = param.data.to("cuda", non_blocking=True)
            if param.grad is not None:
                param.grad = param.grad.to("cuda", non_blocking=True)
        for _, buffer in self.model.named_buffers():
            buffer.data = buffer.data.to("cuda", non_blocking=True)
        # Backload optimizer states to GPU
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    state = self.optimizer.state[param]
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.data = value.data.to("cuda", non_blocking=True)

    def init_fsdp_model(
        self,
        module,
        device_id: int = 0,
        shard_master_addr: str = None,
        shard_port: int = None,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        replicate_master_addr: str | None = None,
        replicate_port: int | None = None,
        replicate_rank: int | None = None,
        replicate_world_size: int | None = None,
    ):
        from functools import partial

        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from orz.exp_engine.parallels.orz_distributed_c10d import orz_init_process_group

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{shard_master_addr}:{shard_port}",
            rank=shard_rank,
            world_size=shard_world_size,
        )
        shard_pg = dist.group.WORLD
        process_group = shard_pg
        sharding_strategy = ShardingStrategy.FULL_SHARD
        replicate_pg = None

        if replicate_master_addr and replicate_port:
            group_name = f"replicate_pg_{replicate_master_addr}_{replicate_port}"
            replicate_pg = orz_init_process_group(
                backend="nccl",
                init_method=f"tcp://{replicate_master_addr}:{replicate_port}",
                rank=replicate_rank,
                world_size=replicate_world_size,
                group_name=group_name,
            )
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
            process_group = (shard_pg, replicate_pg)

        if hasattr(module.model, "get_decoder"):
            transformer_layer_cls = module.model.get_decoder().layers[0].__class__
        elif hasattr(module.model, "layers"):
            transformer_layer_cls = module.model.layers[0].__class__
        else:
            raise ValueError("Cannot automatically find transformer layer class")

        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})

        model = FSDP(
            module.float(),
            device_id=device_id,
            process_group=process_group,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float16
            ),
        )

        return model, shard_pg, replicate_pg

    def save_safetensors(
        self,
        state_dict: dict,
        path: str,
        min_block_size_in_GB: int = 5,
        max_block_size_in_GB: int = 20,
    ):
        from pathlib import Path

        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        total_size = 0
        for param in state_dict.values():
            total_size += param.numel() * param.element_size()

        size_per_block = total_size // 4 + 1
        size_per_block = min(size_per_block, max_block_size_in_GB * 1024**3)
        size_per_block = max(size_per_block, min_block_size_in_GB * 1024**3)

        partial_state_dict = {}
        num_saved_files = 0
        partial_state_dict_size = 0
        safetensors_path = path / "model-1.safetensors"
        meta_info = {"metadata": {"total_size": total_size}, "weight_map": {}}

        for name, param in tqdm(state_dict.items(), total=len(state_dict), desc="Saving weights"):
            partial_state_dict[name] = param
            partial_state_dict_size += param.numel() * param.element_size()
            meta_info["weight_map"][name] = safetensors_path.name
            if partial_state_dict_size > size_per_block:
                save_file(partial_state_dict, safetensors_path, metadata={"format": "pt"})
                num_saved_files += 1
                safetensors_path = path / f"model-{num_saved_files + 1}.safetensors"

                partial_state_dict = {}
                partial_state_dict_size = 0

        if len(partial_state_dict) > 0:
            save_file(partial_state_dict, safetensors_path, metadata={"format": "pt"})

        with open(path / "model.safetensors.index.json", "w") as f:
            json.dump(meta_info, f)

    def iter_fsdp_state_dict(self, model):
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            for k, v in model.state_dict().items():
                gathered_value = torch.empty(
                    v.shape,
                    device=torch.device("cuda"),
                    dtype=v.dtype,
                )
                v.gather(out=gathered_value if dist.get_rank() == 0 else None)
                dist.broadcast(gathered_value, src=0)
                yield k, gathered_value

    def save_model(self, save_path: str, tokenizer, iteration, model_name: str):
        from torch.distributed.fsdp import FullStateDictConfig

        target_path = os.path.join(save_path, f"iter{iteration}", model_name)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = self.model.state_dict()

            if self.global_rank == 0:
                self.save_safetensors(
                    state_dict,
                    target_path,
                    min_block_size_in_GB=5,
                    max_block_size_in_GB=20,
                )
                tokenizer.save_pretrained(target_path)
                self.model_hf_config.save_pretrained(target_path)
        torch.distributed.barrier()

    def all_reduce(self, data, process_group, world_size, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, process_group, world_size, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= world_size
            dist.all_reduce(data, group=process_group, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data


@ray.remote(num_gpus=1)
class FsdpCriticRayActor(BaseFsdpRayActor, CriticRayActorBase):
    def init_model_from_pretrained(self, pretrain, cfg=None):
        import torch.optim as optim
        from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
        from transformers import AutoConfig, AutoModel

        self.args = cfg

        with torch.device("meta"):
            AutoModel.from_config(AutoConfig.from_pretrained(pretrain))
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=self.args.normalize_reward,
            use_flash_attention_2=self.args.flash_attn,
            bf16=self.args.bf16,
            load_in_4bit=self.args.load_in_4bit,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules,
            lora_dropout=self.args.lora_dropout,
            value_head_prefix=self.args.value_head_prefix,
            init_value_head=self.args.pretrain == self.args.critic_pretrain,
            packing_samples=self.args.packing_samples,
        )

        if self.args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_use_reentrant}
            )

        self.model, self.shard_pg, self.replicate_pg = self.init_fsdp_model(
            critic,
            device_id=0,
            shard_master_addr=self.shard_master_addr,
            shard_port=self.shard_port,
            shard_rank=self.shard_rank,
            shard_world_size=self.shard_world_size,
            replicate_master_addr=self.replicate_master_addr,
            replicate_port=self.replicate_port,
            replicate_rank=self.replicate_rank,
            replicate_world_size=self.replicate_world_size,
        )

        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            fused=True,
            lr=cfg.critic_learning_rate,
            betas=cfg.adam_betas,
            weight_decay=cfg.l2,
        )

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (((step + 1) / cfg.num_warmup_steps) if step < cfg.num_warmup_steps else 1.0),
        )

        self.model_hf_config = critic.model.config

        self.critic_loss_fn = ValueLoss(self.args.value_clip)

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
                disable=self.global_rank != 0,
            )
            for local_step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, global_steps, local_step, accumulation_steps)
                critic_update_steps += 1

                # for DP
                # status = self.strategy.all_reduce(status)
                status = self.all_reduce(status, self.shard_pg, self.shard_world_size, op="sum")
                denominator = self.shard_world_size
                if self.replicate_pg:
                    status = self.all_reduce(status, self.replicate_pg, self.replicate_world_size, op="sum")
                    denominator *= self.replicate_world_size
                for k in status:
                    status[k] /= denominator

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

    def training_step(self, experience, global_steps, local_step, accumulation_steps):
        from openrlhf.models.utils import masked_mean

        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = torch.cat(experience.num_actions, dim=0).long().tolist()
            packed_seq_lens = torch.cat(experience.packed_seq_lens, dim=0).long().tolist()
            attention_mask = torch.cat(experience.attention_mask, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.model(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.args.aux_loss_coef > 1e-8:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        loss = loss / accumulation_steps
        loss.backward()
        if (local_step + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.lr_scheduler.get_last_lr()[0],
        }
        return status


@ray.remote(num_gpus=1)
class FsdpPolicyRayActor(BaseFsdpRayActor, PolicyRayActorBase):
    def init_model_from_pretrained(self, pretrain, cfg=None):
        import torch.optim as optim

        self.args = cfg

        actor = Actor(
            pretrain,
            use_flash_attention_2=self.args.flash_attn,
            bf16=self.args.bf16,
            load_in_4bit=self.args.load_in_4bit,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules,
            lora_dropout=self.args.lora_dropout,
            packing_samples=self.args.packing_samples,
        )

        if self.args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_use_reentrant}
            )

        self.model, self.shard_pg, self.replicate_pg = self.init_fsdp_model(
            actor,
            device_id=0,
            shard_master_addr=self.shard_master_addr,
            shard_port=self.shard_port,
            shard_rank=self.shard_rank,
            shard_world_size=self.shard_world_size,
            replicate_master_addr=self.replicate_master_addr,
            replicate_port=self.replicate_port,
            replicate_rank=self.replicate_rank,
            replicate_world_size=self.replicate_world_size,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            fused=True,
            lr=cfg.actor_learning_rate,
            betas=cfg.adam_betas,
            weight_decay=cfg.l2,
        )

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (((step + 1) / cfg.num_warmup_steps) if step < cfg.num_warmup_steps else 1.0),
        )

        self.ema_model = None

        self.model_hf_config = actor.model.config

        # set ppo loss function
        self.actor_loss_fn = PolicyLoss(self.args.eps_clip)

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

        # only different from base class is here
        update_steps = self.args.policy_update_steps

        accumulation_steps = max(1, len(dataloader) // update_steps)

        status_list = []
        status_mean = {}
        policy_update_steps = 0
        for epoch in range(self.args.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Actor Train epoch [{epoch + 1}/{self.args.max_epochs}]",
                disable=self.global_rank != 0,
            )
            for local_step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, global_steps, local_step, accumulation_steps)
                policy_update_steps += 1

                # for DP
                status = self.all_reduce(status, self.shard_pg, self.shard_world_size, op="sum")
                denominator = self.shard_world_size
                if self.replicate_pg:
                    status = self.all_reduce(status, self.replicate_pg, self.replicate_world_size, op="sum")
                    denominator *= self.replicate_world_size
                for k in status:
                    status[k] /= denominator

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

    def training_step(self, experience, global_steps, local_step, accumulation_steps):
        from openrlhf.models.utils import masked_mean

        self.model.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = torch.cat(experience.num_actions, dim=0).long().tolist()
            packed_seq_lens = torch.cat(experience.packed_seq_lens, dim=0).long().tolist()
            attention_mask = torch.cat(experience.attention_mask, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            base_action_log_probs = experience.base_action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

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

        # mixtral
        if self.args.aux_loss_coef > 1e-8:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0

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

        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.args.kl_loss_coef
        loss = loss / accumulation_steps
        loss.backward()

        # ptx loss is not implemented in this version
        if (local_step + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            from loguru import logger

            logger.info("Updated the policy model.")

        # status
        status = {
            "policy_loss": actor_loss.item(),
            "actor_lr": self.lr_scheduler.get_last_lr()[0],
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

    def _init_vllm_engines_actor_group(self, vllm_engines=None):
        pass

    def _broadcast_to_vllm_cudaipc(self, vllm_engines):
        from orz.exp_engine.parallels.orz_distributed_c10d import CUDAIPCHandle

        torch.cuda.empty_cache()
        for name, param in self.iter_fsdp_state_dict(self.model):
            if not isinstance(vllm_engines[0], list):
                raise ValueError("Use vllm_impl=oone!")
            if not name.startswith("model."):
                continue
            x = self.global_rank // self.args.vllm_tensor_parallel_size
            y = self.global_rank % self.args.vllm_tensor_parallel_size
            ref = vllm_engines[x][y].update_weight_internal_with_cuda_ipc.remote(
                name[6:],
                cudaipc_handler=CUDAIPCHandle.from_tensor(param.data),
            )
            ray.get(ref)


@ray.remote(num_gpus=1)
class FSDPRefRayActor(BaseFsdpRayActor):
    def init_model_from_pretrained(self, pretrain, cfg=None):
        from openrlhf.models.actor import Actor

        self.args = cfg
        torch.cuda.set_device("cuda:0")

        actor = Actor(
            pretrain,
            use_flash_attention_2=self.args.flash_attn,
            bf16=self.args.bf16,
            load_in_4bit=self.args.load_in_4bit,
            packing_samples=self.args.packing_samples,
        )

        if cfg.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": cfg.gradient_checkpointing_use_reentrant}
            )

        self.model, self.shard_pg, self.replicate_pg = self.init_fsdp_model(
            actor,
            device_id=0,
            shard_master_addr=self.shard_master_addr,
            shard_port=self.shard_port,
            shard_rank=self.shard_rank,
            shard_world_size=self.shard_world_size,
            replicate_master_addr=self.replicate_master_addr,
            replicate_port=self.replicate_port,
            replicate_rank=self.replicate_rank,
            replicate_world_size=self.replicate_world_size,
        )

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


class FsdpPPORayActorGroup(PPORayActorGroup):
    def __init__(
        self,
        ray_actor_type: type,
        num_total_gpus: int,
        num_shards: int,
        pg: PlacementGroup,
        num_gpus_per_actor: float = 1,
    ) -> None:
        self.ray_actor_type = ray_actor_type
        self._num_total_gpus = num_total_gpus
        self._num_shards = num_shards
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg: PlacementGroup, num_gpus_per_actor: float):
        from collections import defaultdict

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        world_size = self._num_total_gpus
        num_replicate_groups = world_size // self._num_shards

        # Preparing for replicate master ports and address
        used_ports = defaultdict(set)
        if num_replicate_groups > 1:
            replicate_master_addrs, replicate_ports = [], []
            # Hybrid Sharded Data Parallel, need replicate ports
            while len(replicate_ports) < self._num_shards:
                replicate_master_addr, replicate_port = get_master_addr_port_by_bundle(pg, len(replicate_ports))
                if replicate_port not in used_ports[replicate_master_addr]:
                    used_ports[replicate_master_addr].add(replicate_port)
                    replicate_ports.append(replicate_port)
                    replicate_master_addrs.append(replicate_master_addr)

        # Preparing for shard master ports and address
        shard_master_addrs, shard_ports = [], []
        for i in range(num_replicate_groups):
            master_bundle_id = i * self._num_shards
            master_addr, master_port = get_master_addr_port_by_bundle(pg, master_bundle_id)
            while master_port in used_ports[master_addr]:
                _, master_port = get_master_addr_port_by_bundle(pg, master_bundle_id)
            used_ports[master_addr].add(master_port)
            shard_master_addrs.append(master_addr)
            shard_ports.append(master_port)

        # Build actor handlers
        global_rank = 0
        self._actor_handlers = []
        for i in range(num_replicate_groups):
            bundle_id_offset = i * self._num_shards
            shard_master_addr, shard_master_port = shard_master_addrs[i], shard_ports[i]
            for rank in range(self._num_shards):
                env_vars = {
                    "GLOBAL_RANK": global_rank,
                    "GLOBAL_WORLD_SIZE": num_replicate_groups * self._num_shards,
                    "SHARD_MASTER_ADDR": shard_master_addr,
                    "SHARD_PORT": shard_master_port,
                    "SHARD_RANK": rank,
                    "SHARD_WORLD_SIZE": self._num_shards,
                }
                if num_replicate_groups > 1:
                    env_vars.update(
                        {
                            "REPLICATE_MASTER_ADDR": replicate_master_addrs[rank],
                            "REPLICATE_PORT": replicate_ports[rank],
                            "REPLICATE_RANK": i,
                            "REPLICATE_WORLD_SIZE": num_replicate_groups,
                        }
                    )
                env_vars = {k: str(v) for k, v in env_vars.items()}

                actor = self.ray_actor_type.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_id_offset + rank,
                    ),
                    runtime_env={"env_vars": env_vars},
                ).remote()
                self._actor_handlers.append(actor)
                global_rank += 1


class MyTrainer(CustomRewardTrainer):
    async def build_models(self, PolicyRayActor, CriticRayActor, RefRayActor, RewardRayActor=None):
        cfg = self.cfg
        pg = None

        if cfg.colocate_all:
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines * cfg.vllm_tensor_parallel_size
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg

            policy_model = FsdpPPORayActorGroup(
                PolicyRayActor,
                num_total_gpus=cfg.num_total_gpus,
                num_shards=cfg.num_actor_shards,
                pg=pg,
                num_gpus_per_actor=0.2,
            )

            ref_model = FsdpPPORayActorGroup(
                RefRayActor,
                num_total_gpus=cfg.num_total_gpus,
                num_shards=cfg.num_ref_shards,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            if cfg.critic_pretrain:
                critic_model = FsdpPPORayActorGroup(
                    CriticRayActor,
                    num_total_gpus=cfg.num_total_gpus,
                    num_shards=cfg.num_critic_shards,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and not cfg.remote_rm_url and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.2,
                        )
                    )
            else:
                reward_models = None

        else:
            if cfg.colocate_actor_ref:
                assert (
                    cfg.actor_num_nodes == cfg.ref_num_nodes
                    and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [
                    {"GPU": cfg.actor_num_gpus_per_node, "CPU": cfg.actor_num_gpus_per_node}
                    for _ in range(cfg.actor_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )

            # if colocated, create placement group for critic and reward model explicitly.
            pg = None
            if cfg.colocate_critic_reward:
                assert (
                    cfg.critic_num_nodes == cfg.reward_num_nodes
                    and cfg.critic_num_gpus_per_node == cfg.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

                bundles = [
                    {"GPU": cfg.critic_num_gpus_per_node, "CPU": cfg.critic_num_gpus_per_node}
                    for _ in range(cfg.critic_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and not cfg.remote_rm_url and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.25 if pg else 1,
                        )
                    )
            else:
                reward_models = None

        if not cfg.colocate_all:
            refs = []
            refs.extend(ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain, self._max_steps))
            if cfg.critic_pretrain:
                refs.extend(
                    critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain, self._max_steps)
                )
            if not cfg.remote_rm_url and cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    refs.extend(reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))
            await asyncio.gather(*refs)
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
        else:
            await asyncio.gather(*ref_model.async_init_model_from_pretrained(cfg.pretrain, cfg))
            await ref_model.offload_to_cpu()
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(cfg.pretrain, cfg))
            await policy_model.offload_to_cpu()
            if cfg.critic_pretrain:
                await asyncio.gather(*critic_model.async_init_model_from_pretrained(cfg.critic_pretrain, cfg))
                await critic_model.offload_to_cpu()
            if not cfg.remote_rm_url and cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    await asyncio.gather(*reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))

        self.policy_model = policy_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/ref/critic/reward models done")


class PPOExp(BasePPOExp):
    @cached_property
    def PolicyRayActor(self):
        print("custom policy ray actor")
        return FsdpPolicyRayActor

    @cached_property
    def CriticRayActor(self):
        print("custom critic ray actor")
        return FsdpCriticRayActor

    @cached_property
    def RefRayActor(self):
        print("custom ref ray actor")
        return FSDPRefRayActor

    @cached_property
    def get_colocate_pg(self):
        if self.cfg.vllm_impl == "oone":
            total_size = self.cfg.vllm_num_engines * self.cfg.vllm_tensor_parallel_size
            pg = placement_group([{"GPU": 1, "CPU": 1}] * total_size, strategy="PACK")
            ray.get(pg.ready())
            return pg
        if self.cfg.colocate_all:
            pg = placement_group([{"GPU": 1, "CPU": 1}] * self.cfg.vllm_num_engines, strategy="PACK")
            ray.get(pg.ready())
            return pg
        else:
            return None

    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return MyTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        if DEBUG_MODE:
            dialogues = dialogues[:128]
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
            model_type=self.cfg.model_type,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                for loaded_data_item in loaded_data:
                    # only keep file name, without suffix
                    loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
            model_type=self.cfg.model_type,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    def create_inference_engine(self):
        return create_vllm_engines(
            self.cfg.vllm_num_engines,
            self.cfg.vllm_tensor_parallel_size,
            self.cfg.pretrain,
            self.cfg.seed,
            self.cfg.enable_prefix_caching,
            self.cfg.enforce_eager,
            self.cfg.max_len,
            self.cfg.colocate_all,
            self.cfg.enable_chunked_prefill,
            self.cfg.max_num_batched_tokens,
            self.cfg.gpu_memory_utilization,
            self.cfg.micro_rollout_batch_size,
            self.get_colocate_pg,
            vllm_impl=self.cfg.vllm_impl,
        )


def main():
    from orz.ppo.utils import tee_output

    exp = PPOExp().set_cfg(PPOLlamaExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    with tee_output(os.path.join(exp.cfg.save_path, "training.log")):
        asyncio.run(exp.run())


if __name__ == "__main__":
    main()
