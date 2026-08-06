"""Tests for BPE-aware export functionality (Phase 5)."""

import json
import zipfile
import io
from pathlib import Path

import pytest

from backend.export import (
    build_script,
    build_notebook,
    _get_tokenizer_type,
    _get_tokenizer_artifact_path,
    _build_char_tokenizer_code,
    _build_bpe_tokenizer_code,
)


# ── Fixtures ──

@pytest.fixture
def char_config():
    """Config with char tokenizer (default)."""
    return {
        "name": "Char Tokenizer Test",
        "template": "transformer",
        "model": {
            "vocab_size": 65,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.0003,
            "max_iters": 100,
            "eval_interval": 10,
            "eval_iters": 2,
        },
        "data": {
            "dataset": "tiny_shakespeare",
            "tokenizer": "char",
        },
    }


@pytest.fixture
def bpe_1k_config():
    """Config with BPE 1k tokenizer."""
    return {
        "name": "BPE 1K Tokenizer Test",
        "template": "transformer",
        "model": {
            "vocab_size": 1024,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.0003,
            "max_iters": 100,
            "eval_interval": 10,
            "eval_iters": 2,
        },
        "data": {
            "dataset": "tiny_shakespeare",
            "tokenizer": "bpe_1k",
            "tokenizer_artifact": "tiny-shakespeare-bpe-1k-v1.json",
            "vocab_size": 1024,
        },
    }


@pytest.fixture
def bpe_4k_config():
    """Config with BPE 4k tokenizer."""
    return {
        "name": "BPE 4K Tokenizer Test",
        "template": "transformer",
        "model": {
            "vocab_size": 4096,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.0003,
            "max_iters": 100,
            "eval_interval": 10,
            "eval_iters": 2,
        },
        "data": {
            "dataset": "tiny_shakespeare",
            "tokenizer": "bpe_4k",
            "tokenizer_artifact": "tiny-shakespeare-bpe-4k-v1.json",
            "vocab_size": 4096,
        },
    }


# ── Helper function tests ──

class TestTokenizerHelpers:
    """Test tokenizer extraction helper functions."""

    def test_get_tokenizer_type_char(self, char_config):
        """Default/char tokenizer should be 'char'."""
        assert _get_tokenizer_type(char_config) == "char"

    def test_get_tokenizer_type_bpe_1k(self, bpe_1k_config):
        """BPE 1k config should return 'bpe_1k'."""
        assert _get_tokenizer_type(bpe_1k_config) == "bpe_1k"

    def test_get_tokenizer_type_bpe_4k(self, bpe_4k_config):
        """BPE 4k config should return 'bpe_4k'."""
        assert _get_tokenizer_type(bpe_4k_config) == "bpe_4k"

    def test_get_tokenizer_type_missing_data_block(self):
        """Missing data block should default to 'char'."""
        config = {"model": {}, "training": {}}
        assert _get_tokenizer_type(config) == "char"

    def test_get_tokenizer_artifact_path_char(self, char_config):
        """Char config should return None."""
        assert _get_tokenizer_artifact_path(char_config) is None

    def test_get_tokenizer_artifact_path_bpe_1k(self, bpe_1k_config):
        """BPE 1k config should return artifact filename."""
        assert _get_tokenizer_artifact_path(bpe_1k_config) == "tiny-shakespeare-bpe-1k-v1.json"

    def test_get_tokenizer_artifact_path_bpe_4k(self, bpe_4k_config):
        """BPE 4k config should return artifact filename."""
        assert _get_tokenizer_artifact_path(bpe_4k_config) == "tiny-shakespeare-bpe-4k-v1.json"


# ── Tokenizer code generation tests ──

class TestTokenizerCodeGeneration:
    """Test tokenizer code snippet generation."""

    def test_char_tokenizer_code(self):
        """Char tokenizer code should set up stoi/itos from text."""
        from config.settings import settings
        code = _build_char_tokenizer_code(settings)

        assert "sorted(set(text))" in code
        assert "stoi" in code
        assert "itos" in code
        assert "encode = lambda s: [stoi[c] for c in s]" in code

    def test_bpe_tokenizer_code(self, bpe_1k_config):
        """BPE tokenizer code should use Tokenizer.from_file."""
        code = _build_bpe_tokenizer_code(bpe_1k_config, "tiny-shakespeare-bpe-1k-v1.json")

        assert "from tokenizers import Tokenizer" in code
        assert 'Tokenizer.from_file("tiny-shakespeare-bpe-1k-v1.json")' in code
        assert "vocab_size = 1024" in code
        assert "tok.encode(s).ids" in code
        assert "tok.decode(ids)" in code

    def test_bpe_tokenizer_code_vocab_size(self, bpe_4k_config):
        """BPE tokenizer code should use correct vocab_size."""
        code = _build_bpe_tokenizer_code(bpe_4k_config, "tiny-shakespeare-bpe-4k-v1.json")

        assert "vocab_size = 4096" in code


# ── Script generation tests ──

class TestScriptGeneration:
    """Test complete script generation."""

    def test_char_script_generation(self, char_config):
        """Char export should generate unchanged script."""
        script = build_script(char_config)

        # Should have char vocab reconstruction
        assert "sorted(set(text))" in script
        assert "stoi" in script
        assert "itos" in script
        assert "encode = lambda s: [stoi[c] for c in s]" in script

        # Should NOT have tokenizers import
        assert "from tokenizers import Tokenizer" not in script

        # Should have transformer boilerplate
        assert "class TinyTransformerLM" in script
        assert "torch.optim.AdamW" in script

    def test_bpe_script_generation(self, bpe_1k_config):
        """BPE export should use Tokenizer.from_file."""
        script = build_script(bpe_1k_config)

        # Should have BPE-specific code
        assert "from tokenizers import Tokenizer" in script
        assert 'Tokenizer.from_file("tiny-shakespeare-bpe-1k-v1.json")' in script
        assert "vocab_size = 1024" in script
        assert "tok.encode(s).ids" in script

        # Should NOT have char vocab code
        assert "stoi = {ch: i for i, ch in enumerate(chars)}" not in script

        # Should have transformer boilerplate
        assert "class TinyTransformerLM" in script

    def test_bpe_script_has_artifact_path_note(self, bpe_1k_config):
        """BPE script should note that artifact file is required."""
        script = build_script(bpe_1k_config)

        assert "tokenizer artifact" in script or "artifact file" in script

    def test_script_header_includes_tokenizer_type(self, bpe_1k_config):
        """Script header should indicate tokenizer type."""
        script = build_script(bpe_1k_config)

        # Header includes tokenizer info
        assert "Tokenizer: bpe_1k" in script or "tokenizer" in script.split("\n")[2]


# ── Notebook generation tests ──

class TestNotebookGeneration:
    """Test notebook generation."""

    def test_char_notebook_generation(self, char_config):
        """Char notebook should work."""
        notebook_json = build_notebook(char_config)

        # Should be valid JSON
        nb = json.loads(notebook_json)

        # Should have cells
        assert "cells" in nb
        assert len(nb["cells"]) > 0

        # Check for char vocab code somewhere in the notebook
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) > 0

        all_code = "\n".join("".join(c["source"]) for c in code_cells)
        assert "sorted(set(text))" in all_code

    def test_bpe_notebook_generation(self, bpe_1k_config):
        """BPE notebook should include Tokenizer import."""
        notebook_json = build_notebook(bpe_1k_config)

        # Should be valid JSON
        nb = json.loads(notebook_json)

        # Should have cells
        assert "cells" in nb
        assert len(nb["cells"]) > 0

        # Check code cells for BPE code
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) > 0
        all_code = "\n".join("".join(c["source"]) for c in code_cells)

        assert "from tokenizers import Tokenizer" in all_code
        assert 'Tokenizer.from_file("tiny-shakespeare-bpe-1k-v1.json")' in all_code


# ── Edge cases ──

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_config_missing_data_block_defaults_to_char(self):
        """Config without data block should default to char."""
        config = {
            "name": "Test",
            "template": "transformer",
            "model": {
                "vocab_size": 65,
                "block_size": 128,
                "n_embd": 192,
                "n_head": 6,
                "n_layer": 4,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.0003,
                "max_iters": 100,
                "eval_interval": 10,
            },
        }
        script = build_script(config)

        # Should use char vocab, not tokenizers
        assert "sorted(set(text))" in script
        assert "from tokenizers import Tokenizer" not in script

    def test_bpe_config_without_artifact(self, bpe_1k_config):
        """BPE config without artifact filename should still generate code."""
        bpe_1k_config["data"].pop("tokenizer_artifact")

        script = build_script(bpe_1k_config)

        # Should still use BPE approach (with default filename)
        assert "from tokenizers import Tokenizer" in script
        assert "Tokenizer.from_file" in script

    def test_rope_with_bpe(self, bpe_1k_config):
        """BPE should work with RoPE positional encoding."""
        bpe_1k_config["model"]["pos_encoding"] = "rope"

        script = build_script(bpe_1k_config)

        # Should have both RoPE and BPE code
        assert "class RotaryPositionalEncoding" in script
        assert "from tokenizers import Tokenizer" in script
        assert 'Tokenizer.from_file("tiny-shakespeare-bpe-1k-v1.json")' in script
