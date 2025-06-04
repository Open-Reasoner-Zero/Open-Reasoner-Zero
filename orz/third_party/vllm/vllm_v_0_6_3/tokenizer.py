from typing import Optional

from transformers import PreTrainedTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.utils import LRUCache


class TokenizerGroup(TokenizerGroup):
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(
        self, tokenizer: PreTrainedTokenizer, enable_lora: bool, max_num_seqs: int, max_input_length: Optional[int]
    ):
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = tokenizer
        self.lora_tokenizers = LRUCache[PreTrainedTokenizer](capacity=max_num_seqs) if enable_lora else None

    # FIXME(sgm): for simplicity, we assign the special token here
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
