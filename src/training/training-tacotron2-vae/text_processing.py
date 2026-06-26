"""
Portuguese text processing for Tacotron 2 VAE using Phonemes (Gruut).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any

from num2words import num2words
from gruut import sentences

# Exact NVIDIA Tacotron 2 symbols (148 symbols)
_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_arpabet = ['@AA', '@AA0', '@AA1', '@AA2', '@AE', '@AE0', '@AE1', '@AE2', '@AH', '@AH0', '@AH1', '@AH2', '@AO', '@AO0', '@AO1', '@AO2', '@AW', '@AW0', '@AW1', '@AW2', '@AY', '@AY0', '@AY1', '@AY2', '@B', '@CH', '@D', '@DH', '@EH', '@EH0', '@EH1', '@EH2', '@ER', '@ER0', '@ER1', '@ER2', '@EY', '@EY0', '@EY1', '@EY2', '@F', '@G', '@HH', '@IH', '@IH0', '@IH1', '@IH2', '@IY', '@IY0', '@IY1', '@IY2', '@JH', '@K', '@L', '@M', '@N', '@NG', '@OW', '@OW0', '@OW1', '@OW2', '@OY', '@OY0', '@OY1', '@OY2', '@P', '@R', '@S', '@SH', '@T', '@TH', '@UH', '@UH0', '@UH1', '@UH2', '@UW', '@UW0', '@UW1', '@UW2', '@V', '@W', '@Y', '@Z', '@ZH']

NVIDIA_SYMBOLS: List[str] = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

# PT-BR IPA Phonemes (from Gruut)
_pt_ipa = [
    'b', 'd', 'e', 'ej', 'ew', 'f', 'i', 'iw', 'j', 'k', 'l', 'm', 'n', 'o', 'oj', 'ow', 'p', 's', 't', 'u', 'uj', 'v', 'w', 'z', 
    'õ', 'õj̃', 'ĩ', 'ũ', 'ũj̃', 'ɐ', 'ɐj', 'ɐw', 'ɐ̃', 'ɐ̃w̃', 'ɔ', 'ɛ', 'ɛw', 'ɡ', 'ɲ', 'ɹ', 'ɾ', 'ʁ', 'ʃ', 'ʎ', 'ʒ', 'ẽ', 'ẽj̃',
    '|', '‖'  # pauses
]
_pt_ipa_symbols = [f'@pt_{p}' for p in _pt_ipa]

_end = '~'

DEFAULT_SYMBOLS: List[str] = NVIDIA_SYMBOLS + _pt_ipa_symbols + [_end]


# ---- ROBUST PORTUGUESE NORMALIZER ----

_abbreviations = [
    (re.compile(r'\b%s\b\.?' % x[0], re.IGNORECASE), x[1]) for x in [
        ('sr', 'senhor'), ('sra', 'senhora'), ('srta', 'senhorita'),
        ('dr', 'doutor'), ('dra', 'doutora'),
        ('prof', 'professor'), ('profa', 'professora'),
        ('gov', 'governador'), ('gen', 'general'), ('eng', 'engenheiro'),
        ('cel', 'coronel'), ('cap', 'capitão'), ('sgt', 'sargento'), ('ten', 'tenente'),
        ('vol', 'volume'), ('pag', 'página'), ('pág', 'página'),
        ('cia', 'companhia'), ('ltda', 'limitada'),
        ('av', 'avenida'), ('rod', 'rodovia'),
        ('km', 'quilômetros'), ('kg', 'quilos'), ('cm', 'centímetros'), ('mm', 'milímetros'),
        ('min', 'minutos'), ('seg', 'segundos'), ('h', 'horas'),
    ]
]

def expand_abbreviations(text: str) -> str:
    text = re.sub(r'(\d+)(h|min|seg|km|kg|cm|mm)\b', r'\1 \2', text)
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

_currency_re = re.compile(r'R\$\s*(\d+)[,\.](\d{2})', re.IGNORECASE)
def _expand_currency(m: Any) -> str:
    reais = int(m.group(1))
    centavos = int(m.group(2))
    reais_str = num2words(reais, lang='pt-BR') + (" real" if reais == 1 else " reais")
    centavos_str = " e " + num2words(centavos, lang='pt-BR') + (" centavo" if centavos == 1 else " centavos") if centavos > 0 else ""
    return reais_str + centavos_str

_currency_re2 = re.compile(r'R\$\s*(\d+)', re.IGNORECASE)
def _expand_currency2(m: Any) -> str:
    reais = int(m.group(1))
    return num2words(reais, lang='pt-BR') + (" real" if reais == 1 else " reais")

_ordinals_re = re.compile(r'(\d+)[ºª]')
def _expand_ordinal(m: Any) -> str:
    is_fem = 'ª' in m.group(0)
    w = num2words(int(m.group(1)), lang='pt-BR', to='ordinal')
    if is_fem:
        w = w.replace('o ', 'a ').replace('o', 'a')
    return w

_number_re = re.compile(r'\d+')
def _expand_number(m: Any) -> str:
    return num2words(int(m.group(0)), lang='pt-BR')

def portuguese_phonetic_cleaners(text: str) -> str:
    """
    Robust text normalizer + G2P phonemizer using Gruut.
    Returns a space-separated string of phonetic symbols (e.g., "@pt_k @pt_a @pt_z @pt_a").
    """
    text = text.lower()
    text = expand_abbreviations(text)
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_currency_re2, _expand_currency2, text)
    text = re.sub(_ordinals_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    
    # Convert to phonemes using gruut
    phoneme_list = []
    for sent in sentences(text, lang='pt'):
        for i, word in enumerate(sent):
            if word.phonemes:
                for p in word.phonemes:
                    phoneme_list.append(f"@pt_{p}")
                # Add space if not the last word and next word is not punctuation
                if i < len(sent) - 1 and sent[i+1].text not in _punctuation:
                    phoneme_list.append("_space_")
            elif word.text in _punctuation:
                phoneme_list.append(word.text)
                if i < len(sent) - 1 and sent[i+1].text not in _punctuation:
                    phoneme_list.append("_space_")
                
    return " ".join(phoneme_list)

_en_number_re = re.compile(r'\d+')

def english_cleaners(text: str) -> str:
    """Basic English cleaner: lowercase + number expansion. Character-level output."""
    text = text.lower().strip()
    text = re.sub(_en_number_re, lambda m: num2words(int(m.group(0))), text)
    return text


CLEANERS: Dict[str, Any] = {
    "portuguese_phonetic_cleaners": portuguese_phonetic_cleaners,
    "english_cleaners": english_cleaners,
}

# ---- TEXT PROCESSOR CLASS ----

class TextProcessor:
    def __init__(self, symbols: Optional[Sequence[str]] = None, cleaner_names: Optional[Sequence[str]] = None) -> None:
        self.symbols: List[str] = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
        self.cleaner_names: List[str] = list(cleaner_names or ["portuguese_phonetic_cleaners"])
        self._symbol_to_id: Dict[str, int] = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self._id_to_symbol: Dict[int, str] = {idx: symbol for idx, symbol in enumerate(self.symbols)}

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
            cleaner = CLEANERS.get(name)
            if cleaner:
                text = cleaner(text)
        return text

    def text_to_sequence(self, text: str) -> List[int]:
        cleaned_text = self.clean_text(text)
        
        if "phonetic" in self.cleaner_names[0]:
            # Expect space-separated phonemes/symbols
            # We map the special '_space_' token back to a literal space string so split() doesn't swallow it
            symbols = [sym if sym != "_space_" else " " for sym in cleaned_text.split()]
        else:
            # Fallback for plain character cleaners
            symbols = list(cleaned_text)
            
        sequence: List[int] = [self._symbol_to_id[sym] for sym in symbols if sym in self._symbol_to_id]
        sequence.append(self.eos_id)
        return sequence

    def sequence_to_text(self, sequence: Sequence[int]) -> str:
        chars: List[str] = []
        for symbol_id in sequence:
            symbol: Optional[str] = self._id_to_symbol.get(symbol_id)
            if symbol is None or symbol in {_pad, _end}:
                continue
            chars.append(symbol)
        return " ".join(chars) if "phonetic" in self.cleaner_names[0] else "".join(chars)

    def save(self, path: Path) -> None:
        path = Path(path)
        payload: Dict[str, Any] = {
            "symbols": self.symbols,
            "cleaner_names": self.cleaner_names,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TextProcessor:
        payload: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(symbols=payload["symbols"], cleaner_names=payload["cleaner_names"])

def build_symbols_from_texts(texts: Sequence[str]) -> List[str]:
    return DEFAULT_SYMBOLS
