"""Code view + export endpoints."""

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from backend import db
from backend.export import build_script, build_notebook
from config.settings import settings

router = APIRouter(prefix="/api/code", tags=["code"])

TEMPLATE_DIRS = {
    "transformer": Path(__file__).parent.parent / "training" / "templates" / "transformer",
    "moe": Path(__file__).parent.parent / "training" / "templates" / "moe",
    # nanogpt has model.py but no data.py — it reuses the transformer's data
    # path. The loop below skips files that don't exist, so that's fine.
    "nanogpt": Path(__file__).parent.parent / "training" / "templates" / "nanogpt",
    "rnn": Path(__file__).parent.parent / "training" / "templates" / "rnn",
}

TEMPLATE_FILES = ["model.py", "data.py"]

# transformer / moe / nanogpt all train on the SAME data pipeline — only the
# model differs. moe used to fake its own data.py with a symlink to the
# transformer's, which dangled inside the trainer container (absolute host
# path) and told users moe had its own loader when it does not. Symlink
# removed 2026-09-16; templates without their own copy of a file now borrow
# the transformer's, labelled so nobody mistakes it for theirs.
SHARED_SOURCE_TEMPLATE = "transformer"

BACKEND_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

SHARED_FILES = {
    "runner.py": BACKEND_ROOT / "training" / "runner.py",
    "presets.py": PROJECT_ROOT / "config" / "presets.py",
    "settings.py": PROJECT_ROOT / "config" / "settings.py",
}


async def _get_config(experiment_id: int, run_id: int | None = None) -> dict:
    """Config to export.

    Prefer the RUN's config_snapshot when we have one: that is what actually
    trained. The experiment's own config may have been edited since the run, so
    exporting it would hand the user a script that does not reproduce their run.
    Falls back to the experiment config for a never-run experiment.
    """
    if run_id is not None:
        run = await db.get_training_run(run_id)
        if run and run.get("config_snapshot"):
            return json.loads(run["config_snapshot"])
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return json.loads(exp["config_json"])


@router.get("/{experiment_id}")
async def get_template_code(experiment_id: int):
    """Return actual template source files for code view panel."""
    config = await _get_config(experiment_id)
    template_key = config.get("template", "transformer")
    template_dir = TEMPLATE_DIRS.get(template_key)
    if template_dir is None:
        raise HTTPException(404, f"Template '{template_key}' not found")

    files = {}
    shared_dir = TEMPLATE_DIRS[SHARED_SOURCE_TEMPLATE]
    for filename in TEMPLATE_FILES:
        filepath = template_dir / filename
        if filepath.exists():
            files[filename] = filepath.read_text()
            continue
        shared = shared_dir / filename
        if template_key != SHARED_SOURCE_TEMPLATE and shared.exists():
            files[f"{filename} (shared)"] = shared.read_text()

    for label, filepath in SHARED_FILES.items():
        if filepath.exists():
            files[label] = filepath.read_text()

    return {"experiment_id": experiment_id, "template": template_key, "files": files}


@router.get("/{experiment_id}/export.py")
async def export_script(experiment_id: int, run_id: int | None = None):
    """Download standalone .py training script with config baked in."""
    config = await _get_config(experiment_id, run_id)
    script = build_script(config)
    return PlainTextResponse(
        script,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.py"},
    )


@router.get("/{experiment_id}/export.ipynb")
async def export_notebook(experiment_id: int, run_id: int | None = None):
    """Download .ipynb notebook with config baked in."""
    config = await _get_config(experiment_id, run_id)
    notebook_json = build_notebook(config)
    return PlainTextResponse(
        notebook_json,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.ipynb"},
    )


@router.get("/{experiment_id}/export.zip")
async def export_bundle(experiment_id: int, run_id: int | None = None):
    """Download full export bundle: script, notebook, config, notes, and
    (if run_id given) the run's metrics — see docs §6.2 export bundle spec.
    For BPE tokenizers, includes the tokenizer artifact JSON file.
    """
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    # Same rule as the .py/.ipynb endpoints: the run's snapshot wins when given.
    config = await _get_config(experiment_id, run_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("export.py", build_script(config))
        zf.writestr("export.ipynb", build_notebook(config))
        zf.writestr("config.json", json.dumps(config, indent=2))
        zf.writestr("notes.md", exp.get("notes_md") or "")

        # Include tokenizer artifact if BPE-based
        tokenizer_type = config.get("data", {}).get("tokenizer", "char")
        if tokenizer_type != "char":
            artifact_filename = config.get("data", {}).get("tokenizer_artifact")
            if artifact_filename:
                artifact_path = settings.data_dir / "tokenizers" / artifact_filename
                if artifact_path.exists():
                    zf.writestr(artifact_filename, artifact_path.read_text())

        if run_id is not None:
            metrics_file = settings.data_dir / "runs" / str(run_id) / "metrics.jsonl"
            if metrics_file.exists():
                zf.writestr("metrics.jsonl", metrics_file.read_text())

    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.zip"},
    )
