from dataclasses import dataclass
from omegaconf.listconfig import ListConfig
from typing import Optional

@dataclass
class CommonPPOConfig:
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B"
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = "orz_ckpt/default"
    save_path: str = "orz_ckpt/default"
    tensorboard_log_dir: str = "orz_logs/default"
    prompt_data: ListConfig = ListConfig(["data/orz_math_57k_collected.json"])
    eval_prompt_data: ListConfig = ListConfig([
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ])
    prompt_data_probs: ListConfig = ListConfig([1.0])
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True
    num_episodes: int = 20
    rollout_batch_size: int = 128
    n_samples_per_prompt: int = 64
    micro_rollout_batch_size: int = 128
    policy_update_steps: int = 1
    critic_update_steps: int = 12
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True
    enable_eval: bool = True
    eval_interval: int = 10
    generate_max_len: int = 8000
    max_len: int = 8192
    packing_max_len: int = 16384
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])
