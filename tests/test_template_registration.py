"""Every trainable template must be registered everywhere it is consumed.

Real bug, 2026-09-16: nanoGPT was added to TEMPLATE_REGISTRY and to the
training dispatcher, but nobody added it to codegen's TEMPLATE_DIRS. The
/api/code/{id} endpoint 404'd for every nanoGPT experiment, which in turn
blanked the whole bottom panel in the UI (the Metrics column lives inside
the same component and was hidden along with the code).

These tests exist so the NEXT template can't be half-registered the same way.
"""

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


def test_every_template_has_an_export_script_builder():
    """Closed 2026-09-16: nanogpt and moe now reuse _char_lm_script."""
    missing = set(TEMPLATE_REGISTRY) - set(SCRIPT_BUILDERS)
    assert not missing, f"templates missing from SCRIPT_BUILDERS: {sorted(missing)}"


def test_templates_without_their_own_data_py_borrow_the_transformer_s():
    """transformer / moe / nanogpt share one data pipeline; only the model differs.

    moe used to fake its own data.py with a symlink to the transformer's. That
    symlink used an ABSOLUTE host path, so it dangled inside the trainer
    container, and it implied moe had a loader of its own. Removed 2026-09-16;
    the code panel now borrows the transformer's copy and labels it '(shared)'
    so nobody mistakes it for the template's own file.
    """
    from backend.api.codegen import SHARED_SOURCE_TEMPLATE, TEMPLATE_DIRS, TEMPLATE_FILES

    shared_dir = TEMPLATE_DIRS[SHARED_SOURCE_TEMPLATE]
    for filename in TEMPLATE_FILES:
        assert (shared_dir / filename).exists(), (
            f"{SHARED_SOURCE_TEMPLATE} must carry {filename} — it is the fallback "
            "every other template borrows from"
        )

    # no template may reintroduce a symlink to dodge this
    for key, path in TEMPLATE_DIRS.items():
        for filename in TEMPLATE_FILES:
            candidate = path / filename
            assert not candidate.is_symlink(), (
                f"{key}/{filename} is a symlink — these dangle inside the trainer "
                "container (absolute host paths). Use the shared-source fallback."
            )
