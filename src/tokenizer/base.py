from pathlib import Path
from typing import List, Dict
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    def __init__(self, vocab: List[str]) -> None:
        self.idx2tok = self._create_idx2tok(tokens=vocab)
        self.tok2idx = {tok: idx for idx, tok in self.idx2tok.items()}
    
    @abstractmethod
    def _create_idx2tok(self, tokens: List[str]) -> Dict[int, str]:
        pass
    
    def __len__(self) -> int:
        return len(self.tok2idx)
    
    def encode(self, text: str, max_len: int = 100) -> List[int]: 
        tokens = (
            [self.tok2idx.get("<sos>")]
            + [self.tok2idx.get(ch) for ch in text if ch in self.tok2idx]
            + [self.tok2idx.get("<eos>")]
        )
        tokens += [self.tok2idx.get("<pad>")] * (max_len - len(tokens))
        return tokens[:max_len]
    
    def decode(self, indices: list[int]) -> str:
        return "".join(
            self.idx2tok[idx]
            for idx in indices
            if idx in self.idx2tok and idx not in {
                self.tok2idx["<pad>"],
                self.tok2idx["<sos>"],
                self.tok2idx["<eos>"]
            }
        )
    
    @classmethod
    def from_token_file(cls, path: str | Path) -> "Tokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path.as_posix()}")
        tokenizer = cls(vocab=[])
        tokenizer.idx2tok = enum_str(path.read_text(encoding="utf-8"))
        tokenizer.tok2idx = {tok: idx for idx, tok in tokenizer.idx2tok.items()}
        
        return tokenizer

    def to_token_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True) 
        tokens = [self.idx2tok[i] for i in range(len(self))]
        path.write_text("\n".join(tokens), encoding="utf-8")
    
    
def enum_str(s: str, start: int = 0) -> dict[int, str]:
    return {i: tok for i, tok in enumerate(s.strip().splitlines(), start=start)}
