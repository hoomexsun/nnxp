from pathlib import Path


def save_pairs(pairs: list[tuple], path: str | Path, sep:str="\t") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"{x}{sep}{y}" for x, y in pairs]), encoding="utf-8")
    
    
def load_pairs(path: str | Path, sep:str="\t") -> tuple[list, list]:
    xs, ys = [], []
    for line in Path(path).read_text(encoding="utf-8").strip().split("\n"):
        x,  y, *_ = line.strip().split(sep, maxsplit=1)
        xs.append(x)
        ys.append(y)
    return xs, ys
