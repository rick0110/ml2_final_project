"""Text tokenization utilities for Phase 3 training."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoTokenizer


class SubwordTextTokenizer:
    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.hf_tokenizer.pad_token_id is None:
            self.hf_tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_idx = int(self.hf_tokenizer.pad_token_id)
        self.eos_idx = self._resolve_eos_id()

    def _resolve_eos_id(self) -> int:
        if self.hf_tokenizer.eos_token_id is not None:
            return int(self.hf_tokenizer.eos_token_id)
        if self.hf_tokenizer.sep_token_id is not None:
            return int(self.hf_tokenizer.sep_token_id)
        if self.hf_tokenizer.cls_token_id is not None:
            return int(self.hf_tokenizer.cls_token_id)
        return self.pad_idx

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.hf_tokenizer.encode(
            text.strip(),
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return int(self.hf_tokenizer.vocab_size)


class BatchTextTokenizer:
    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256):
        self.tokenizer = SubwordTextTokenizer(model_name=model_name, max_length=max_length)
        self.max_length = max_length
        self.pad_idx = self.tokenizer.pad_idx
        self.eos_idx = self.tokenizer.eos_idx

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer.hf_tokenizer(
            [text.strip() for text in texts],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(dtype=torch.long)

    def encode_batch_with_attention_mask(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer.hf_tokenizer(
            [text.strip() for text in texts],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(dtype=torch.long),
            "attention_mask": encoded["attention_mask"].to(dtype=torch.long),
        }
