"""Load tokenizers by ID — registry pattern."""

from pathlib import Path

from backend.training.templates.transformer.data import load_tiny_shakespeare
from backend.training.tokenizers.base import TokenizerProtocol
from backend.training.tokenizers.char import CharTokenizer
from backend.training.tokenizers.bpe import BPETokenizer
from config.settings import settings


def load_tokenizer(data_config: dict) -> TokenizerProtocol:
    """Load a tokenizer by ID from the data config.

    Args:
        data_config: Dict with at minimum a "tokenizer" key (string ID).
                     May also contain "tokenizer_artifact" (path override).

    Returns:
        A tokenizer instance (CharTokenizer, BPETokenizer, etc.).

    Raises:
        ValueError: If the tokenizer ID is unknown.
    """
    tokenizer_id = data_config.get("tokenizer", "char")

    if tokenizer_id == "char":
        # Build char tokenizer from Tiny Shakespeare
        text = load_tiny_shakespeare()
        return CharTokenizer.from_text(text)

    elif tokenizer_id == "bpe_1k":
        # Load pre-built 1k BPE tokenizer
        artifact_path = data_config.get(
            "tokenizer_artifact",
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json",
        )
        return BPETokenizer(artifact_path)

    elif tokenizer_id == "bpe_4k":
        # Load pre-built 4k BPE tokenizer
        artifact_path = data_config.get(
            "tokenizer_artifact",
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-4k-v1.json",
        )
        return BPETokenizer(artifact_path)

    else:
        raise ValueError(
            f"Unknown tokenizer ID: {tokenizer_id!r}. "
            f"Supported: 'char', 'bpe_1k', 'bpe_4k'."
        )
