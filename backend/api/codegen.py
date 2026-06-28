"""Code view + export endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from backend import db
from backend.export import build_script, build_notebook

router = APIRouter(prefix="/api/code", tags=["code"])

TEMPLATE_DIRS = {
    "transformer": Path(__file__).parent.parent / "training" / "templates" / "transformer",
    "moe": Path(__file__).parent.parent / "training" / "templates" / "moe",
    "rnn": Path(__file__).parent.parent / "training" / "templates" / "rnn",
}

TEMPLATE_FILES = ["model.py", "data.py"]

BACKEND_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

SHARED_FILES = {
    "runner.py": BACKEND_ROOT / "training" / "runner.py",
    "presets.py": PROJECT_ROOT / "config" / "presets.py",
    "settings.py": PROJECT_ROOT / "config" / "settings.py",
    "export.py": BACKEND_ROOT / "export.py",
    "main.py": BACKEND_ROOT / "main.py",
    "db.py": BACKEND_ROOT / "db.py",
}


async def _get_config(experiment_id: int) -> dict:
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
    for filename in TEMPLATE_FILES:
        filepath = template_dir / filename
        if filepath.exists():
            files[filename] = filepath.read_text()

    for label, filepath in SHARED_FILES.items():
        if filepath.exists():
            files[label] = filepath.read_text()

    return {"experiment_id": experiment_id, "template": template_key, "files": files}


@router.get("/{experiment_id}/export.py")
async def export_script(experiment_id: int):
    """Download standalone .py training script with config baked in."""
    config = await _get_config(experiment_id)
    script = build_script(config)
    return PlainTextResponse(
        script,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.py"},
    )


@router.get("/{experiment_id}/export.ipynb")
async def export_notebook(experiment_id: int):
    """Download .ipynb notebook with config baked in."""
    config = await _get_config(experiment_id)
    notebook_json = build_notebook(config)
    return PlainTextResponse(
        notebook_json,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.ipynb"},
    )
