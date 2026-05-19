"""Text processing utilities for TTS training.

This module provides subword tokenization utilities (BPE/WordPiece/SentencePiece)
using Hugging Face tokenizers.
"""

from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


class SubwordTextTokenizer:
    """Robust subword tokenizer for Portuguese and multilingual text.

    By default this uses `xlm-roberta-base` (SentencePiece-style subword model),
    which works well.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.hf_tokenizer.pad_token_id is None:
            self.hf_tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_idx = self.hf_tokenizer.pad_token_id
        self.eos_idx = self._resolve_eos_id()

    def _resolve_eos_id(self) -> int:
        """Resolve EOS token id even when the tokenizer has no explicit EOS."""
        if self.hf_tokenizer.eos_token_id is not None:
            return self.hf_tokenizer.eos_token_id
        if self.hf_tokenizer.sep_token_id is not None:
            return self.hf_tokenizer.sep_token_id
        if self.hf_tokenizer.cls_token_id is not None:
            return self.hf_tokenizer.cls_token_id
        return self.pad_idx

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs with truncation."""
        normalized_text = text.strip()
        token_ids = self.hf_tokenizer.encode(
            normalized_text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.max_length,
        )
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return int(self.hf_tokenizer.vocab_size)


class BatchTextTokenizer:
    """Tokenize text batches with subword tokenization and fixed-length padding."""

    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256):
        self.tokenizer = SubwordTextTokenizer(model_name=model_name, max_length=max_length)
        self.max_length = max_length
        self.pad_idx = self.tokenizer.pad_idx
        self.eos_idx = self.tokenizer.eos_idx

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a batch of texts.

        Returns token IDs as tensor with shape (batch_size, max_length).
        """
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
        """Tokenize a batch and return both IDs and attention mask."""
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

    def decode_batch(self, batch_ids: torch.Tensor) -> List[str]:
        """Decode token ID tensor into text strings."""
        decoded_texts = []
        for seq_ids in batch_ids:
            decoded_texts.append(self.tokenizer.decode(seq_ids.tolist()))
        return decoded_texts


if __name__ == "__main__":
    print("Testing SubwordTextTokenizer:")
    tokenizer = SubwordTextTokenizer()

    text = "Olá, como você está?"
    token_ids = tokenizer.encode(text)
    print(f"  Input: {text}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Decoded: {tokenizer.decode(token_ids)}")
    print(f"  Vocabulary size: {len(tokenizer)}")

    print("\nTesting BatchTextTokenizer:")
    batch_tokenizer = BatchTextTokenizer(max_length=64)
    texts = ["Este é um teste.", "Outro texto de exemplo."]
    batch_tensor = batch_tokenizer.encode_batch(texts)
    print(f"  Batch shape: {batch_tensor.shape}")
    print(f"  Decoded: {batch_tokenizer.decode_batch(batch_tensor)}")
