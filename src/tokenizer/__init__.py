from pathlib import Path
from typing import Iterable

from ..data.utils import load_pairs

from .char_tokenizer import CharTokenizer
# from .phn_tokenizer import PhonemeTokenizer

TOKENIZER_REGISTRY = dict(
    char=CharTokenizer,
    # phn=PhonemeTokenizer
)

def _infer_token_type(tokens_file: Path) -> str:
    parts = tokens_file.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer token type from file name: {tokens_file.name}")
    return parts[-2]

def build_tokenizer(xs: Iterable[str], tokens_file: Path) -> None:
    token_type = _infer_token_type(tokens_file)
    tokenizer_cls = TOKENIZER_REGISTRY.get(token_type)
    if tokenizer_cls is None:
        raise ValueError(f"Unsupported token type: {token_type}")
    
    tokenizer = tokenizer_cls(xs)
    tokenizer.to_token_file(tokens_file)

def load_tokenizer(tokens_file: Path):
    if not tokens_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokens_file.as_posix()}")
    
    token_type = _infer_token_type(tokens_file)
    tokenizer_cls = TOKENIZER_REGISTRY.get(token_type)
    if tokenizer_cls is None:
        raise ValueError(f"Unsupported token type: {token_type}")
    
    return tokenizer_cls.from_token_file(tokens_file)

def prepare_tokenizers(x_tokens_file, y_tokens_file, db_file):
    xs, ys = [], []
    try:
        x_tokenizer = load_tokenizer(x_tokens_file)
        y_tokenizer = load_tokenizer(y_tokens_file)
    except FileNotFoundError:
        xs, ys = load_pairs(db_file)
        build_tokenizer(xs, x_tokens_file)
        build_tokenizer(ys, y_tokens_file)
        x_tokenizer = load_tokenizer(x_tokens_file)
        y_tokenizer = load_tokenizer(y_tokens_file)
    return x_tokenizer, y_tokenizer, xs, ys
