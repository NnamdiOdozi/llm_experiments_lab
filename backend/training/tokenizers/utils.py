"""Utilities for tokenizer management and validation."""

from pathlib import Path

import torch


def assert_tokenizer_matches(checkpoint_path: str, config: dict) -> None:
    """Guard against resuming with mismatched tokenizers.

    If the checkpoint was trained with a different tokenizer than the one
    in the current config, raise an error — embedding shapes differ and
    the model cannot load its state_dict correctly.

    Args:
        checkpoint_path: Path to the checkpoint file.
        config: Current training config with data block.

    Raises:
        ValueError: If tokenizer ID or vocab_size mismatch is detected.
    """
    if not Path(checkpoint_path).exists():
        return  # No checkpoint to check

    # Load only the metadata from checkpoint (no model weights)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cp_tokenizer_id = cp.get("tokenizer_id", "char")
    cp_vocab_size = cp.get("vocab_size")

    # Get expected tokenizer from config
    config_data = config.get("data", {"tokenizer": "char"})
    config_tokenizer_id = config_data.get("tokenizer", "char")
    config_vocab_size = config_data.get("vocab_size")

    # Check tokenizer ID mismatch
    if cp_tokenizer_id != config_tokenizer_id:
        raise ValueError(
            f"Tokenizer mismatch: checkpoint trained with '{cp_tokenizer_id}' "
            f"but config specifies '{config_tokenizer_id}'. "
            f"Cannot resume — embedding shapes differ."
        )

    # Check vocab_size mismatch (catches BPE variant mismatches)
    if cp_vocab_size is not None and config_vocab_size is not None:
        if cp_vocab_size != config_vocab_size:
            raise ValueError(
                f"Vocab size mismatch: checkpoint has vocab_size={cp_vocab_size} "
                f"but config specifies vocab_size={config_vocab_size}. "
                f"Cannot resume — embedding shapes differ."
            )
