"""Tests for tokenizer abstraction and implementations."""

import tempfile
from pathlib import Path

import pytest
import torch

from backend.training.tokenizers.char import CharTokenizer
from backend.training.tokenizers.bpe import BPETokenizer
from backend.training.tokenizers.loader import load_tokenizer
from backend.training.tokenizers.dataset import TokenizedTextDataset
from config.settings import settings


class TestCharTokenizer:
    """Tests for character-level tokenizer."""

    def test_from_text_tiny_shakespeare(self):
        """CharTokenizer.from_text on Tiny Shakespeare matches 65-char vocab."""
        from backend.training.templates.transformer.data import load_tiny_shakespeare
        text = load_tiny_shakespeare()
        tokenizer = CharTokenizer.from_text(text)
        assert tokenizer.vocab_size == 65

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding should be lossless."""
        sample = "hello world\ntest"
        tokenizer = CharTokenizer.from_text(sample)
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample

    def test_id_to_token_regular_char(self):
        """id_to_token returns correct fields for a regular character."""
        tokenizer = CharTokenizer.from_text("abc")
        token_meta = tokenizer.id_to_token(0)  # 'a'
        assert token_meta["raw"] == "a"
        assert token_meta["display"] == "a"
        assert token_meta["decoded"] == "a"

    def test_id_to_token_newline(self):
        """id_to_token makes newline visible in display."""
        tokenizer = CharTokenizer.from_text("a\nb")
        # Find the newline token ID
        ids = tokenizer.encode("\n")
        token_meta = tokenizer.id_to_token(ids[0])
        assert token_meta["raw"] == "\n"
        assert token_meta["display"] == "⏎"
        assert token_meta["decoded"] == "\n"

    def test_id_to_token_space(self):
        """id_to_token makes space visible in display."""
        tokenizer = CharTokenizer.from_text("a b")
        ids = tokenizer.encode(" ")
        token_meta = tokenizer.id_to_token(ids[0])
        assert token_meta["raw"] == " "
        assert token_meta["display"] == "␠"
        assert token_meta["decoded"] == " "

    def test_id_to_token_tab(self):
        """id_to_token makes tab visible in display."""
        tokenizer = CharTokenizer.from_text("a\tb")
        ids = tokenizer.encode("\t")
        token_meta = tokenizer.id_to_token(ids[0])
        assert token_meta["raw"] == "\t"
        assert token_meta["display"] == "␉"
        assert token_meta["decoded"] == "\t"


class TestBPETokenizer:
    """Tests for BPE tokenizer with pre-built artifacts."""

    def test_load_bpe_1k_artifact(self):
        """Load the pre-built 1k BPE tokenizer."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        assert tokenizer.vocab_size == 1024

    def test_load_bpe_4k_artifact(self):
        """Load the pre-built 4k BPE tokenizer."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-4k-v1.json"
        )
        assert tokenizer.vocab_size == 4096

    def test_bpe_encode_decode_roundtrip_1k(self):
        """Encoding and decoding BPE 1k should be lossless with spaces and newlines."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        sample = "King Henry:\nTo be, or not to be."
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample, f"Round-trip failed: {sample!r} -> {decoded!r}"

    def test_bpe_encode_decode_roundtrip_4k(self):
        """Encoding and decoding BPE 4k should be lossless with spaces and newlines."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-4k-v1.json"
        )
        sample = "King Henry:\nTo be, or not to be."
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample, f"Round-trip failed: {sample!r} -> {decoded!r}"

    def test_bpe_id_to_token_returns_dict(self):
        """id_to_token returns dict with raw/display/decoded fields."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        token_meta = tokenizer.id_to_token(0)
        assert isinstance(token_meta, dict)
        assert "raw" in token_meta
        assert "display" in token_meta
        assert "decoded" in token_meta

    def test_bpe_id_to_token_byte_level_space_marker(self):
        """id_to_token makes byte-level space marker Ġ visible as ␠."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        # Find a token with Ġ (if one exists in the vocab)
        # This is a soft test — just verify the replacement happens if present
        for token_id in range(min(100, tokenizer.vocab_size)):
            token_meta = tokenizer.id_to_token(token_id)
            # No Ġ should appear in display (it should be replaced with ␠)
            assert "Ġ" not in token_meta["display"]

    def test_bpe_id_to_token_space_prefixed_decoded_starts_with_space(self):
        """id_to_token for space-prefixed token: display shows ␠, decoded starts with space."""
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        # Encode text with spaces to get space-prefixed tokens
        sample = "King Henry"
        encoded = tokenizer.encode(sample)
        # Decode each token and look for one that decodes to a space
        for token_id in encoded:
            token_meta = tokenizer.id_to_token(token_id)
            # If the decoded starts with a space, display should start with ␠
            if token_meta["decoded"].startswith(" "):
                assert token_meta["display"].startswith("␠"), (
                    f"Token {token_id}: display={token_meta['display']!r} "
                    f"should start with ␠ when decoded starts with space"
                )


class TestTokenizedTextDataset:
    """Tests for the unified dataset class."""

    def test_init_with_char_tokenizer(self):
        """Initialize dataset with char tokenizer."""
        sample = "hello world"
        tokenizer = CharTokenizer.from_text(sample)
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=4,
            batch_size=2,
        )
        assert dataset.vocab_size == tokenizer.vocab_size

    def test_init_with_bpe_tokenizer(self):
        """Initialize dataset with BPE tokenizer."""
        sample = "hello world this is a test"
        tokenizer = BPETokenizer(
            settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json"
        )
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=4,
            batch_size=2,
        )
        assert dataset.vocab_size == 1024

    def test_train_val_split(self):
        """90/10 train/val split is correct."""
        sample = "a" * 100
        tokenizer = CharTokenizer.from_text(sample)
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=4,
            batch_size=2,
        )
        # Total encoded length should be ~100
        total = len(dataset.train_data) + len(dataset.val_data)
        assert total == 100
        # Train should be ~90, val ~10
        assert len(dataset.train_data) == 90
        assert len(dataset.val_data) == 10

    def test_get_batch_shape(self):
        """get_batch returns correct tensor shapes."""
        sample = "hello world this is a test sentence for batching"
        tokenizer = CharTokenizer.from_text(sample)
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=5,
            batch_size=3,
        )
        x, y = dataset.get_batch("train")
        assert x.shape == (3, 5)
        assert y.shape == (3, 5)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_get_batch_val_split(self):
        """get_batch can sample from val split."""
        # Use longer sample to ensure val split has enough tokens
        sample = "hello world this is a test sentence " * 10
        tokenizer = CharTokenizer.from_text(sample)
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=4,
            batch_size=2,
        )
        x_val, y_val = dataset.get_batch("val")
        assert x_val.shape == (2, 4)
        assert y_val.shape == (2, 4)

    def test_dataset_encode_decode(self):
        """Dataset delegates encode/decode to tokenizer."""
        sample = "test"
        tokenizer = CharTokenizer.from_text(sample)
        dataset = TokenizedTextDataset(
            tokenizer=tokenizer,
            text=sample,
            block_size=2,
            batch_size=1,
        )
        encoded = dataset.encode("test")
        decoded = dataset.decode(encoded)
        assert decoded == "test"


class TestLoader:
    """Tests for the tokenizer loader registry."""

    def test_load_char_tokenizer(self):
        """load_tokenizer with 'char' ID."""
        tokenizer = load_tokenizer({"tokenizer": "char"})
        assert isinstance(tokenizer, CharTokenizer)
        assert tokenizer.vocab_size == 65  # Tiny Shakespeare

    def test_load_bpe_1k_tokenizer(self):
        """load_tokenizer with 'bpe_1k' ID."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_1k"})
        assert isinstance(tokenizer, BPETokenizer)
        assert tokenizer.vocab_size == 1024

    def test_load_bpe_4k_tokenizer(self):
        """load_tokenizer with 'bpe_4k' ID."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_4k"})
        assert isinstance(tokenizer, BPETokenizer)
        assert tokenizer.vocab_size == 4096

    def test_load_unknown_tokenizer_raises(self):
        """load_tokenizer with unknown ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tokenizer ID"):
            load_tokenizer({"tokenizer": "unknown"})

    def test_load_with_artifact_override(self):
        """load_tokenizer accepts custom tokenizer_artifact path."""
        tokenizer = load_tokenizer({
            "tokenizer": "bpe_1k",
            "tokenizer_artifact": settings.data_dir / "tokenizers" / "tiny-shakespeare-bpe-1k-v1.json",
        })
        assert isinstance(tokenizer, BPETokenizer)
        assert tokenizer.vocab_size == 1024
