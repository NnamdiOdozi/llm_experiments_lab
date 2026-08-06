"""Unified dataset class for any tokenizer — encodes text once at init."""

import torch

from backend.training.tokenizers.base import TokenizerProtocol


class TokenizedTextDataset:
    """Dataset that encodes text once at initialization and caches as a tensor.

    Replaces CharDataset for any tokenizer (char, BPE, etc.).
    90/10 train/val split with identical get_batch() semantics to CharDataset.
    """

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        text: str,
        block_size: int,
        batch_size: int,
    ):
        """Initialize dataset with a tokenizer and raw text.

        Args:
            tokenizer: A tokenizer implementing TokenizerProtocol.
            text: Raw text to encode and split into train/val.
            block_size: Context window size for language modeling.
            batch_size: Batch size for get_batch().
        """
        self.tokenizer = tokenizer
        self.text = text
        self.block_size = block_size
        self.batch_size = batch_size

        # Encode text once and cache
        ids = self.tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)

        # 90/10 train/val split
        n_train = int(0.9 * len(data))
        self.train_data = data[:n_train]
        self.val_data = data[n_train:]

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer's vocabulary size."""
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text (delegates to tokenizer)."""
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode IDs (delegates to tokenizer)."""
        return self.tokenizer.decode(ids)

    def get_batch(
        self, split: str, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch from train or val split.

        Args:
            split: "train" or "val".
            device: Torch device to move tensors to.

        Returns:
            Tuple of (x, y) where:
            - x: shape (batch_size, block_size) — input token IDs.
            - y: shape (batch_size, block_size) — target token IDs (shifted by 1).
        """
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
