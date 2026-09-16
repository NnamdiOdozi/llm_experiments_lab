"""Byte-for-byte snapshots of the exported standalone training scripts.

Purpose: `_transformer_script` is being refactored into one shared
`_char_lm_script` so that nanogpt and moe stop 500-ing on export (see
tests/test_template_registration.py). Only `transformer` had any export test
before that work, so there was nothing to catch a silent change to its output
during the refactor. These goldens are that safety net.

The configs are pinned in tests/fixtures/export/*_config.json rather than read
from lab.db, so the goldens can't drift with the user's live database.

If a change to the generated script is INTENDED, regenerate the goldens with:
    .venv/bin/python -m tests.regen_export_goldens
and review the diff in the commit.
"""

import json
from pathlib import Path

import pytest

from backend.export import build_script

FIXTURES = Path(__file__).parent / "fixtures" / "export"
SNAPSHOT_TEMPLATES = ["transformer", "rnn", "nanogpt", "moe"]


def _load(key: str) -> tuple[dict, str]:
    config = json.loads((FIXTURES / f"{key}_config.json").read_text())
    golden = (FIXTURES / f"{key}_script.py.golden").read_text()
    return config, golden


@pytest.mark.parametrize("key", SNAPSHOT_TEMPLATES)
def test_generated_script_matches_golden(key):
    config, golden = _load(key)
    assert build_script(config) == golden, (
        f"{key} export output changed. If intentional, regenerate the golden "
        f"(see this module's docstring) and review the diff."
    )


@pytest.mark.parametrize("key", SNAPSHOT_TEMPLATES)
def test_generated_script_is_valid_python(key):
    """A script we hand a user must at least parse."""
    config, _ = _load(key)
    compile(build_script(config), f"<{key}_export>", "exec")


def test_scaled_init_guard_expects_the_right_projection_count():
    """The exported nanoGPT scales residual projections by matching parameter
    NAME SUFFIXES ('out_proj.weight', 'net.2.weight') — the same technique
    Karpathy uses ('c_proj.weight'). But 'net.2' is POSITIONAL inside
    FeedForward's nn.Sequential, so reordering that Sequential would silently
    skip the init. The generated script guards against that by counting matches
    and expecting 2 per layer (attention out_proj + FeedForward net.2).

    This pins the naming convention the guard depends on, against the lab's own
    model, so a rename here fails loudly rather than in a user's download.
    """
    from config.presets import PRESETS
    from backend.training.templates import TEMPLATE_REGISTRY

    preset = PRESETS["nanogpt"]
    model = TEMPLATE_REGISTRY["nanogpt"]["build_model"](preset)
    matched = sum(
        1 for name, _ in model.named_parameters()
        if name.endswith("out_proj.weight") or name.endswith("net.2.weight")
    )
    expected = 2 * preset["model"]["n_layer"]
    assert matched == expected, (
        f"scaled-init naming drifted: matched {matched}, expected {expected}. "
        "The exported script's guard uses this same rule."
    )


def test_nanogpt_export_carries_grad_clip_and_guard():
    config, script = _load("nanogpt")[0], None
    script = build_script(config)
    assert "GRAD_CLIP = 1.0" in script, "nanoGPT export lost Karpathy's clip value"
    assert "clip_grad_norm_" in script, "nanoGPT export does not clip"
    assert "_scaled != 2 * N_LAYER" in script, "scaled-init guard missing"
