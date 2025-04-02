import torch
from vllm.core.scheduler import Scheduler


class LLMActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from .vllm_worker_wrap import OffloadableVLLMWorker

            vllm.worker.worker.Worker = OffloadableVLLMWorker
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs[
                    "worker_cls"
                ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap.OffloadableVLLMWorker"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs[
                            "worker_module_name"
                        ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "OffloadableVLLMWorker"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        kwargs["enforce_eager"] = True
        self.llm = vllm.LLM(*args, **kwargs)
        self.scheduler_config = self.llm.llm_engine.scheduler_config
        self.model_config = self.llm.llm_engine.model_config
        self.cache_config = self.llm.llm_engine.cache_config
        self.lora_config = self.llm.llm_engine.lora_config
        self.parallel_config = self.llm.llm_engine.parallel_config

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def get_ip_and_port(self):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.get_ip_and_port()
        else:
            return self.llm.llm_engine.model_executor._run_workers("get_ip_and_port")

    def offload_to_cpu(self):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.offload_cpu()
        else:
            return self.llm.llm_engine.model_executor._run_workers("offload_cpu")

    def backload_to_gpu(self):
        if self.use_gpu_executor:
            self.llm.llm_engine.model_executor.driver_worker.load_gpu()
        else:
            self.llm.llm_engine.model_executor._run_workers("load_gpu")
        # rebuild scheduler
        self.llm.llm_engine.scheduler = [
            Scheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id] if self.model_config.use_async_output_proc else None,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight_internal_with_cuda_ipc(
                name, dtype, shape, cudaipc_handler, empty_cache
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "update_weight_internal_with_cuda_ipc", name, dtype, shape, cudaipc_handler, empty_cache
            )

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def get_gpu_memory(self):
        """获取当前Actor使用的GPU内存"""
        import torch

        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        for name, param in model_runner.model.named_parameters():
            # 计算关键统计信息
            tensor_stats = {
                "mean": param.mean().item(),
                "std": param.std().item(),
                "norm": param.norm().item(),
                "shape": tuple(param.shape),
                # 可选：计算一些极值
                "max": param.max().item(),
                "min": param.min().item(),
            }
            stats[name] = tensor_stats
        return stats


class ColocateVllmActor:
    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from orz.third_party.vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

        self.llm = LLM(
            model,
            tokenizer,
            model_hf_config=model.config,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            dtype=dtype,
            load_format="hf",
            **kwargs,
        )

        self.scheduler_config = self.llm.llm_engine.scheduler_config
        self.model_config = self.llm.llm_engine.model_config
        self.cache_config = self.llm.llm_engine.cache_config
        self.lora_config = self.llm.llm_engine.lora_config
        self.parallel_config = self.llm.llm_engine.parallel_config

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str,
        backend: str,
    ):
        from orz.exp_engine.parallels.orz_distributed_c10d import orz_init_process_group

        self._model_update_group = orz_init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def get_ip_and_port(self):
        import socket

        import ray

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        return master_address, master_port

    def offload_to_cpu(self):
        assert self.llm.llm_engine.model_config.enforce_eager, "Must use eager mode to offload!"
        model = self.llm.llm_engine.model_executor.worker.model_runner.model
        for param in model.parameters():
            param.meta_tensor = param.data.to("meta")  # Record the dtype and shape
            param.data = torch.Tensor([])

        self.llm.free_cache_engine()
        torch.cuda.empty_cache()

    def init_cache_engine(self):
        self.llm.init_cache_engine()

    def free_cache_engine(self):
        self.llm.free_cache_engine()

    def backload_to_gpu(self, init_vllm_engines: bool = True):
        assert self.llm.llm_engine.model_config.enforce_eager, "Must use eager mode to offload!"
        model = self.llm.llm_engine.model_executor.worker.model_runner.model
        for param in model.parameters():
            if len(param.data) == 0:
                param.data = torch.empty_like(param.meta_tensor, device="cuda")
                param.meta_tensor = None
            else:
                param.data = param.data.to("cuda", non_blocking=True)
        if init_vllm_engines:
            self.llm.init_cache_engine()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

        worker = self.llm.llm_engine.model_executor.worker
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        worker.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def update_weight_internal_with_cuda_ipc(
            self, name, cudaipc_handler, empty_cache=False, dtype=None, shape=None
        ):
        weight = cudaipc_handler.rebuild().clone()
        worker = self.llm.llm_engine.model_executor.worker
        worker.model_runner.model.load_weights(weights=[(name, weight)])
        del weight
        if empty_cache:
            torch.cuda.empty_cache()

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def get_gpu_memory(self):
        """获取当前Actor使用的GPU内存"""
        import torch

        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        for name, param in model_runner.model.named_parameters():
            # 计算关键统计信息
            tensor_stats = {
                "mean": param.mean().item(),
                "std": param.std().item(),
                "norm": param.norm().item(),
                "shape": tuple(param.shape),
                # 可选：计算一些极值
                "max": param.max().item(),
                "min": param.min().item(),
            }
            stats[name] = tensor_stats
        return stats
