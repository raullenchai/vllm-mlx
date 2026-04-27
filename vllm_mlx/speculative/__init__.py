# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding utilities for vllm-mlx.
"""

from .ngram_mod import NGramModDecoder, ngram_mod_generate_step
from .prompt_lookup import PromptLookupDecoder

__all__ = [
    "NGramModDecoder",
    "PromptLookupDecoder",
    "ngram_mod_generate_step",
]
