from pathlib import Path
from typing import List

from .base import Tokenizer

class CharTokenizer(Tokenizer):
    def __init__(self, vocab: List[str]) -> None:
        special_tokens = ["<pad>", "<sos>", "<eos>"]
        tokens = special_tokens + sorted(set("".join(vocab)))
        super().__init__(vocab=tokens) 

    def _create_idx2tok(self, tokens: List[str]) -> dict[int, str]:
        return {idx: tok for idx, tok in enumerate(tokens)}
