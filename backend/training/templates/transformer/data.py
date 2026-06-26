"""Dataset loading for Tiny Shakespeare (character-level)."""

from pathlib import Path

import httpx
import torch

from config.settings import settings

CACHE_FILE = "tiny_shakespeare.txt"


class CharDataset:
    """Character-level dataset with encode/decode and batch sampling."""

    def __init__(self, text: str, block_size: int, batch_size: int):
        self.text = text
        self.block_size = block_size
        self.batch_size = batch_size

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n_train = int(0.9 * len(data))
        self.train_data = data[:n_train]
        self.val_data = data[n_train:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


def load_tiny_shakespeare() -> str:
    """Download Tiny Shakespeare text, caching to disk."""
    cache_path = settings.data_dir / CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_text()

    response = httpx.get(settings.shakespeare_url, timeout=settings.http_timeout)
    response.raise_for_status()
    text = response.text
    cache_path.write_text(text)
    return text
