"""Portuguese text processing for Tacotron2-VAE (character-level, tacotron-style)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

from num2words import num2words

_pad = "_"
_punctuation = "!\'(),.:;? "
_special = "-"
_end = "~"
_letters = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç"
_extra = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_SYMBOLS = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_extra) + [_end]

_whitespace_re = re.compile(r"\s+")
_number_pattern = re.compile(r"\d+")


def _replace_numbers(match):
    return num2words(int(match.group(0)), lang="pt-BR")


def portuguese_cleaners(text: str) -> str:
    text = _number_pattern.sub(_replace_numbers, text)
    replaces = {"–": "-", "—": "-", "−": "-", "·": "", "ı": "õ"}
    for old_char, new_char in replaces.items():
        text = text.replace(old_char, new_char)
    text = text.lower()
    text = re.sub(_whitespace_re, " ", text)
    return text.strip()


CLEANERS = {
    "portuguese_cleaners": portuguese_cleaners,
}


class TextProcessor:
    def __init__(self, symbols: Sequence[str] | None = None, cleaner_names: Sequence[str] | None = None):
        self.symbols = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
        self.cleaner_names = list(cleaner_names or ["portuguese_cleaners"])
        self._symbol_to_id = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self._id_to_symbol = {idx: symbol for idx, symbol in enumerate(self.symbols)}

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def pad_id(self) -> int:
        return self._symbol_to_id[_pad]

    @property
    def eos_id(self) -> int:
        return self._symbol_to_id[_end]

    def clean_text(self, text: str) -> str:
        for name in self.cleaner_names:
            cleaner = CLEANERS[name]
            text = cleaner(text)
        return text

    def text_to_sequence(self, text: str) -> List[int]:
        text = self.clean_text(text)
        sequence = [self._symbol_to_id[char] for char in text if char in self._symbol_to_id]
        sequence.append(self.eos_id)
        return sequence

    def sequence_to_text(self, sequence: Sequence[int]) -> str:
        chars = []
        for symbol_id in sequence:
            symbol = self._id_to_symbol.get(symbol_id)
            if symbol is None or symbol in {_pad, _end}:
                continue
            chars.append(symbol)
        return "".join(chars)

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "symbols": self.symbols,
            "cleaner_names": self.cleaner_names,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TextProcessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(symbols=payload["symbols"], cleaner_names=payload["cleaner_names"])


def build_symbols_from_texts(texts: Sequence[str]) -> List[str]:
    chars = set()
    processor = TextProcessor()
    for text in texts:
        cleaned = processor.clean_text(text)
        chars.update(cleaned)

    ordered = [_pad] + list(_special) + list(_punctuation)
    ordered.extend(sorted(ch for ch in chars if ch not in ordered and ch != _end))
    ordered.append(_end)
    return ordered
