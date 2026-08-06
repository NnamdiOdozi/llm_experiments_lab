"""Phase 2 integration tests: tokenizer selection + config plumbing + resume guard."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from backend.api.experiments import normalize_config
from backend.training.tokenizers.loader import load_tokenizer
from backend.training.tokenizers.dataset import TokenizedTextDataset
from backend.training.tokenizers.utils import assert_tokenizer_matches
from backend.training.templates.transformer.data import load_tiny_shakespeare
from config.settings import settings


class TestNormalizeConfig:
    """Test config normalization for backward compatibility."""

    def test_legacy_config_no_data_block_becomes_char(self):
        """Legacy config without data block normalizes to char/tiny_shakespeare."""
        legacy_config = {
            "template": "transformer",
            "model": {"vocab_size": 65, "block_size": 128},
            "training": {"batch_size": 64},
        }
        normalized = normalize_config(legacy_config)
        assert normalized["data"]["tokenizer"] == "char"
        assert normalized["data"]["dataset"] == "tiny_shakespeare"
        assert normalized["data"]["vocab_size"] == 65

    def test_legacy_rnn_config_becomes_dinos(self):
        """Legacy RNN config normalizes to char/dinos."""
        legacy_config = {
            "template": "rnn",
            "model": {"vocab_size": 29},
            "training": {"epochs": 50},
        }
        normalized = normalize_config(legacy_config, template="rnn")
        assert normalized["data"]["tokenizer"] == "char"
        assert normalized["data"]["dataset"] == "dinos"
        assert normalized["data"]["vocab_size"] == 29

    def test_config_with_data_block_unchanged(self):
        """Config that already has a data block stays unchanged (vocab_size re-derived)."""
        config = {
            "template": "transformer",
            "data": {
                "dataset": "tiny_shakespeare",
                "tokenizer": "char",
                "tokenizer_artifact": None,
                "vocab_size": 65,
            },
            "model": {"vocab_size": 65},
        }
        normalized = normalize_config(config)
        assert normalized["data"]["tokenizer"] == "char"
        assert normalized["data"]["vocab_size"] == 65  # Derived from CharTokenizer

    def test_bpe_1k_config_vocab_size_derived(self):
        """BPE 1k config vocab_size is derived from tokenizer."""
        config = {
            "template": "transformer",
            "data": {
                "dataset": "tiny_shakespeare",
                "tokenizer": "bpe_1k",
                "tokenizer_artifact": None,
            },
            "model": {"vocab_size": 1024},
        }
        normalized = normalize_config(config)
        assert normalized["data"]["tokenizer"] == "bpe_1k"
        assert normalized["data"]["vocab_size"] == 1024

    def test_bpe_4k_config_vocab_size_derived(self):
        """BPE 4k config vocab_size is derived from tokenizer."""
        config = {
            "template": "transformer",
            "data": {
                "dataset": "tiny_shakespeare",
                "tokenizer": "bpe_4k",
                "tokenizer_artifact": None,
            },
            "model": {"vocab_size": 4096},
        }
        normalized = normalize_config(config)
        assert normalized["data"]["tokenizer"] == "bpe_4k"
        assert normalized["data"]["vocab_size"] == 4096


class TestConfigVocabSizePlumbing:
    """Test that vocab_size flows correctly from tokenizer→config→model."""

    def test_char_tokenizer_yields_vocab_65(self):
        """CharTokenizer on Tiny Shakespeare has vocab_size 65."""
        tokenizer = load_tokenizer({"tokenizer": "char"})
        assert tokenizer.vocab_size == 65

    def test_bpe_1k_tokenizer_yields_vocab_1024(self):
        """BPE 1k tokenizer has vocab_size 1024."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_1k"})
        assert tokenizer.vocab_size == 1024

    def test_bpe_4k_tokenizer_yields_vocab_4096(self):
        """BPE 4k tokenizer has vocab_size 4096."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_4k"})
        assert tokenizer.vocab_size == 4096

    def test_tokenized_dataset_vocab_size_matches_tokenizer(self):
        """TokenizedTextDataset vocab_size matches its tokenizer."""
        tokenizer = load_tokenizer({"tokenizer": "char"})
        text = load_tiny_shakespeare()
        dataset = TokenizedTextDataset(tokenizer, text, block_size=128, batch_size=64)
        assert dataset.vocab_size == tokenizer.vocab_size
        assert dataset.vocab_size == 65

    def test_bpe_1k_dataset_vocab_size_1024(self):
        """TokenizedTextDataset with BPE 1k has vocab_size 1024."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_1k"})
        text = load_tiny_shakespeare()
        dataset = TokenizedTextDataset(tokenizer, text, block_size=128, batch_size=64)
        assert dataset.vocab_size == 1024


class TestResumeGuard:
    """Test checkpoint tokenizer matching guards."""

    def test_assert_tokenizer_matches_no_checkpoint(self):
        """Guard passes silently when checkpoint does not exist."""
        config = {"data": {"tokenizer": "char", "vocab_size": 65}}
        # Should not raise
        assert_tokenizer_matches("/nonexistent/checkpoint.pt", config)

    def test_assert_tokenizer_matches_char_char(self):
        """Guard passes when checkpoint and config both use char."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.pt"
            cp = {
                "model_state": {},
                "optimizer_state": {},
                "tokenizer_id": "char",
                "vocab_size": 65,
            }
            torch.save(cp, cp_path)

            config = {"data": {"tokenizer": "char", "vocab_size": 65}}
            # Should not raise
            assert_tokenizer_matches(str(cp_path), config)

    def test_assert_tokenizer_matches_char_to_bpe_raises(self):
        """Guard raises when checkpoint is char but config is BPE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.pt"
            cp = {
                "model_state": {},
                "optimizer_state": {},
                "tokenizer_id": "char",
                "vocab_size": 65,
            }
            torch.save(cp, cp_path)

            config = {"data": {"tokenizer": "bpe_1k", "vocab_size": 1024}}
            with pytest.raises(ValueError, match="Tokenizer mismatch"):
                assert_tokenizer_matches(str(cp_path), config)

    def test_assert_tokenizer_matches_bpe_vocab_size_mismatch_raises(self):
        """Guard raises when checkpoint and config have different vocab sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.pt"
            cp = {
                "model_state": {},
                "optimizer_state": {},
                "tokenizer_id": "bpe_1k",
                "vocab_size": 1024,
            }
            torch.save(cp, cp_path)

            config = {"data": {"tokenizer": "bpe_1k", "vocab_size": 4096}}
            with pytest.raises(ValueError, match="Vocab size mismatch"):
                assert_tokenizer_matches(str(cp_path), config)

    def test_assert_tokenizer_matches_bpe_1k_bpe_1k(self):
        """Guard passes when checkpoint and config both use BPE 1k."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.pt"
            cp = {
                "model_state": {},
                "optimizer_state": {},
                "tokenizer_id": "bpe_1k",
                "vocab_size": 1024,
            }
            torch.save(cp, cp_path)

            config = {"data": {"tokenizer": "bpe_1k", "vocab_size": 1024}}
            # Should not raise
            assert_tokenizer_matches(str(cp_path), config)


class TestTokenizedDatasetBatching:
    """Test that TokenizedTextDataset batching works correctly."""

    def test_get_batch_char_tokenizer(self):
        """get_batch returns correct shapes with char tokenizer."""
        tokenizer = load_tokenizer({"tokenizer": "char"})
        text = load_tiny_shakespeare()
        dataset = TokenizedTextDataset(tokenizer, text, block_size=128, batch_size=64)
        x, y = dataset.get_batch("train")
        assert x.shape == (64, 128)
        assert y.shape == (64, 128)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_get_batch_bpe_1k_tokenizer(self):
        """get_batch returns correct shapes with BPE 1k tokenizer."""
        tokenizer = load_tokenizer({"tokenizer": "bpe_1k"})
        text = load_tiny_shakespeare()
        dataset = TokenizedTextDataset(tokenizer, text, block_size=128, batch_size=64)
        x, y = dataset.get_batch("train")
        assert x.shape == (64, 128)
        assert y.shape == (64, 128)
