from typing import Dict

import torch.nn as nn
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


def update_hf_weight_loader():
    print("no hf weight loader need to be updated")
    return


def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    assert isinstance(actor_weights, Dict)
    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_weights.keys():
            del actor_weights["lm_head.weight"]
        vllm_model.load_weights(actor_weights.items())
    for _, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
        # FIXME: Remove this after Mixtral is updated
        # to use quant_method.
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
    vllm_model = vllm_model.cuda()
