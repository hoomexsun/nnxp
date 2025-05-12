from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch


class XlitDataset(Dataset):
    def __init__(
        self,
        xs: List[str],
        ys: List[str],
        x_tokenizer,
        y_tokenizer,
        max_len: int,
    ) -> None:
        self.xs = xs
        self.ys = ys
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        x, y = self.xs[idx], self.ys[idx]
        x_encoded = self.x_tokenizer.encode(x, self.max_len)
        y_encoded = self.y_tokenizer.encode(y, self.max_len)
        return {
            "input": torch.tensor(x_encoded, dtype=torch.long),
            "target": torch.tensor(y_encoded, dtype=torch.long),
            "input_text": x,
            "target_text": y,
        }

    def save_pairs_to_file(
        self, save_path: Union[str, Path], indices: Optional[List[int]] = None
    ) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            if indices is None:
                iterable = zip(self.xs, self.ys)
            else:
                iterable = ((self.xs[i], self.ys[i]) for i in indices)
            for x, y in iterable:
                f.write(f"{x}\t{y}\n")


def load_dataloaders(
    xs: List[str],
    ys: List[str],
    x_tokenizer,
    y_tokenizer,
    max_len: int = 100,
    batch_size: int = 32,
    val_ratio: float = 0.25,
    train_file: Optional[Union[str, Path]] = None,
    val_file: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset = XlitDataset(xs, ys, x_tokenizer, y_tokenizer, max_len)
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Save data pairs if requested
    if train_file:
        dataset.save_pairs_to_file(train_file, indices=train_dataset.indices)
    if val_file:
        dataset.save_pairs_to_file(val_file, indices=val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
