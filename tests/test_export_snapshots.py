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
SNAPSHOT_TEMPLATES = ["transformer", "rnn"]


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
