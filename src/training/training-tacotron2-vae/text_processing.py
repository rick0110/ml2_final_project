"""
Portuguese text processing for Tacotron 2 VAE.

Responsibilities:
    - Clean and normalize Portuguese text (number expansion, punctuation mapping).
    - Convert text sequences into integer token IDs for model embedding.
    - Handle symbol vocabulary (phonemes/characters) including padding and EOS.
    - Save and load text processor configurations.

Main Classes:
    - TextProcessor: Primary utility for text-to-ID and ID-to-text conversion.

Main Functions:
    - portuguese_cleaners: Normalize raw Portuguese text.
    - build_symbols_from_texts: Generate a unique symbol set from a corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any

from num2words import num2words

# Special symbols
_pad: str = "_"
_punctuation: str = "!\'(),.:;? "
_special: str = "-"
_end: str = "~"
_letters: str = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç"
_extra: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_SYMBOLS: List[str] = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_extra) + [_end]

_whitespace_re: re.Pattern = re.compile(r"\s+")
_number_pattern: re.Pattern = re.compile(r"\d+")


def _replace_numbers(match: Any) -> str:
    """
    Expand numeric digits into words.

    Args:
        match (re.Match): Regex match object.

    Returns:
        str: Text representation of the number.
    """
    return num2words(int(match.group(0)), lang="pt-BR")


def portuguese_cleaners(text: str) -> str:
    """
    Clean and normalize Portuguese text.

    Args:
        text (str): Input raw text.

    Returns:
        str: Cleaned text.
    """
    # Replace numbers with words
    text = _number_pattern.sub(_replace_numbers, text)
    
    # Standardize punctuation and special characters
    replaces: Dict[str, str] = {"–": "-", "—": "-", "−": "-", "·": "", "ı": "õ"}
    for old_char, new_char in replaces.items():
        text = text.replace(old_char, new_char)
    
    text = text.lower()
    text = re.sub(_whitespace_re, " ", text)
    return text.strip()


CLEANERS: Dict[str, Any] = {
    "portuguese_cleaners": portuguese_cleaners,
}


class TextProcessor:
    """
    Processor for character-level tokenization.

    Architecture:
        Cleaning -> Symbol Mapping -> Indexing.

    Inputs:
        text: Raw string.

    Outputs:
        sequence: List of integer token IDs.

    Example:
        >>> processor = TextProcessor()
        >>> ids = processor.text_to_sequence("Olá mundo!")
    """
    def __init__(self, symbols: Optional[Sequence[str]] = None, cleaner_names: Optional[Sequence[str]] = None) -> None:
        """
        Initialize the TextProcessor.

        Args:
            symbols (Optional[Sequence[str]]): List of unique symbols. Defaults to DEFAULT_SYMBOLS.
            cleaner_names (Optional[Sequence[str]]): List of cleaning function names.
        """
        self.symbols: List[str] = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
        self.cleaner_names: List[str] = list(cleaner_names or ["portuguese_cleaners"])
        self._symbol_to_id: Dict[str, int] = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self._id_to_symbol: Dict[int, str] = {idx: symbol for idx, symbol in enumerate(self.symbols)}

    @property
    def n_symbols(self) -> int:
        """int: Total number of symbols in vocabulary."""
        return len(self.symbols)

    @property
    def pad_id(self) -> int:
        """int: ID of the padding symbol."""
        return self._symbol_to_id[_pad]

    @property
    def eos_id(self) -> int:
        """int: ID of the end-of-sentence symbol."""
        return self._symbol_to_id[_end]

    def clean_text(self, text: str) -> str:
        """
        Apply all configured cleaners.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned text.
        """
        for name in self.cleaner_names:
            cleaner = CLEANERS[name]
            text = cleaner(text)
        return text

    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text string to ID sequence.

        Args:
            text (str): Input text.

        Returns:
            List[int]: Sequence of token IDs including EOS.
        """
        text = self.clean_text(text)
        sequence: List[int] = [self._symbol_to_id[char] for char in text if char in self._symbol_to_id]
        sequence.append(self.eos_id)
        return sequence

    def sequence_to_text(self, sequence: Sequence[int]) -> str:
        """
        Convert ID sequence back to text string.

        Args:
            sequence (Sequence[int]): Token IDs.

        Returns:
            str: Reconstructed text string.
        """
        chars: List[str] = []
        for symbol_id in sequence:
            symbol: Optional[str] = self._id_to_symbol.get(symbol_id)
            if symbol is None or symbol in {_pad, _end}:
                continue
            chars.append(symbol)
        return "".join(chars)

    def save(self, path: Path) -> None:
        """
        Save processor configuration to JSON.

        Args:
            path (Path): Destination file path.
        """
        path = Path(path)
        payload: Dict[str, Any] = {
            "symbols": self.symbols,
            "cleaner_names": self.cleaner_names,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TextProcessor:
        """
        Load processor configuration from JSON.

        Args:
            path (Path): Source file path.

        Returns:
            TextProcessor: Initialized processor.
        """
        payload: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(symbols=payload["symbols"], cleaner_names=payload["cleaner_names"])


def build_symbols_from_texts(texts: Sequence[str]) -> List[str]:
    """
    Extract unique symbols from a list of texts.

    Args:
        texts (Sequence[str]): Raw input texts.

    Returns:
        List[str]: Ordered list of unique symbols.
    """
    chars = set()
    processor: TextProcessor = TextProcessor()
    for text in texts:
        cleaned: str = processor.clean_text(text)
        chars.update(cleaned)

    ordered: List[str] = [_pad] + list(_special) + list(_punctuation)
    ordered.extend(sorted(ch for ch in chars if ch not in ordered and ch != _end))
    ordered.append(_end)
    return ordered
