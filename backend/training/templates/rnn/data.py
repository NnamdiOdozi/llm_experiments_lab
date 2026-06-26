"""Dataset loading for RNN character-level LM (dinosaur names)."""

from pathlib import Path

import httpx
import torch

from config.settings import settings

CACHE_FILE = "dinos.txt"


class DinosDataset(torch.utils.data.Dataset):
    """Character-level dataset of dinosaur names with <start> and <end> delimiters."""

    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len

        with open(data_path, "r", encoding="utf-8") as f:
            names = [line.strip().lower() for line in f if line.strip()]

        self.vocab = self._build_vocab(names)
        self.vocab_size = len(self.vocab)
        self.id_to_token = {i: ch for i, ch in enumerate(self.vocab)}
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.corpus = self._build_corpus(names)

    def _build_vocab(self, names: list[str]) -> list[str]:
        all_text = "".join(f"<{name}>" for name in names)
        return sorted(set(all_text))

    def _build_corpus(self, names: list[str]) -> list[int]:
        corpus = []
        for name in names:
            tokens = [self.token_to_id[ch] for ch in f"<{name}>"]
            corpus.extend(tokens)
        return corpus

    def __len__(self):
        return (len(self.corpus) - 1) // self.seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        return (
            torch.tensor(self.corpus[start : start + self.seq_len], dtype=torch.long),
            torch.tensor(self.corpus[start + 1 : start + self.seq_len + 1], dtype=torch.long),
        )


def load_dinos_dataset(seq_len: int = 50) -> DinosDataset:
    """Download dinos.txt if needed and return a DinosDataset."""
    cache_path = settings.data_dir / CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        response = httpx.get(settings.dinos_url, timeout=settings.http_timeout, follow_redirects=True)
        response.raise_for_status()
        cache_path.write_text(response.text)

    return DinosDataset(str(cache_path), seq_len)
