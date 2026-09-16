"""Every trainable template must be registered everywhere it is consumed.

Real bug, 2026-09-16: nanoGPT was added to TEMPLATE_REGISTRY and to the
training dispatcher, but nobody added it to codegen's TEMPLATE_DIRS. The
/api/code/{id} endpoint 404'd for every nanoGPT experiment, which in turn
blanked the whole bottom panel in the UI (the Metrics column lives inside
the same component and was hidden along with the code).

These tests exist so the NEXT template can't be half-registered the same way.
"""

import pytest

from backend.api.codegen import TEMPLATE_DIRS
from backend.export import SCRIPT_BUILDERS
from backend.training.templates import TEMPLATE_REGISTRY
from backend.training.train_worker import TRAIN_DISPATCHERS


def test_every_template_has_a_training_dispatcher():
    missing = set(TEMPLATE_REGISTRY) - set(TRAIN_DISPATCHERS)
    assert not missing, f"templates with no train dispatcher: {sorted(missing)}"


def test_every_template_has_a_code_view_directory():
    """Guards the /api/code/{id} 404 described in the module docstring."""
    missing = set(TEMPLATE_REGISTRY) - set(TEMPLATE_DIRS)
    assert not missing, f"templates missing from codegen TEMPLATE_DIRS: {sorted(missing)}"


def test_code_view_directories_actually_exist_on_disk():
    for key, path in TEMPLATE_DIRS.items():
        assert path.is_dir(), f"TEMPLATE_DIRS['{key}'] points at a missing dir: {path}"
        # model.py is the one file every template is expected to expose;
        # data.py is optional (nanogpt reuses the transformer's data path).
        assert (path / "model.py").exists(), f"{key}: no model.py in {path}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP, 2026-09-16: SCRIPT_BUILDERS only covers transformer and rnn, "
        "so /api/code/{id}/export.py|.ipynb|.zip return HTTP 500 for moe and "
        "nanogpt. MoE has been broken this way since before nanoGPT existed. "
        "Writing correct builders is real work (the transformer builder is ~410 "
        "lines) and is tracked in DEFERRED_WORK.md. When it is fixed, DELETE "
        "this xfail marker — strict=True makes the suite fail on an unexpected "
        "pass, which is the reminder."
    ),
)
def test_every_template_has_an_export_script_builder():
    missing = set(TEMPLATE_REGISTRY) - set(SCRIPT_BUILDERS)
    assert not missing, f"templates missing from SCRIPT_BUILDERS: {sorted(missing)}"
