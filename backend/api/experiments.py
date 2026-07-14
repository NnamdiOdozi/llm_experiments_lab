"""Experiment CRUD endpoints."""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import db
from backend.logging_config import audit_log
from config.presets import PRESETS

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class CreateExperimentRequest(BaseModel):
    name: str
    config: dict
    preset_key: str | None = None


class UpdateNotesRequest(BaseModel):
    notes_md: str


class UpdateConfigRequest(BaseModel):
    config: dict


@router.get("")
async def list_experiments():
    experiments = await db.list_experiments()
    for exp in experiments:
        exp["config"] = json.loads(exp["config_json"])
        del exp["config_json"]
    return experiments


@router.post("")
async def create_experiment(req: CreateExperimentRequest):
    exp_id = await db.create_experiment(req.name, req.config, req.preset_key)
    audit_log.info("Experiment created: id=%d name='%s' preset=%s", exp_id, req.name, req.preset_key)
    return {"id": exp_id, "name": req.name}


@router.post("/from-preset/{preset_key}")
async def create_from_preset(preset_key: str):
    if preset_key not in PRESETS:
        raise HTTPException(404, f"Preset '{preset_key}' not found")
    preset = PRESETS[preset_key]
    exp_id = await db.create_experiment(preset["name"], preset, preset_key)
    audit_log.info("Experiment from preset: id=%d preset='%s' template=%s", exp_id, preset_key, preset["template"])
    return {"experiment_id": exp_id, "name": preset["name"], "config": preset}


@router.get("/presets")
async def list_presets():
    return [{"key": k, **v} for k, v in PRESETS.items()]


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    exp["config"] = json.loads(exp["config_json"])
    del exp["config_json"]
    return exp


@router.patch("/{experiment_id}/config")
async def update_config(experiment_id: int, req: UpdateConfigRequest):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    # Direct user request, 2026-07-15: max_new_tokens must never exceed
    # block_size — applies identically to every template (transformer and
    # MoE both share the same diagnostics/generation code, see
    # docs/DESIGN_DECISIONS.md §57). Single validation point here since
    # ConfigPanel is the only way to edit either field.
    max_new_tokens = req.config.get("inference", {}).get("max_new_tokens")
    block_size = req.config.get("model", {}).get("block_size")
    if (
        isinstance(max_new_tokens, (int, float))
        and isinstance(block_size, (int, float))
        and max_new_tokens > block_size
    ):
        raise HTTPException(
            400,
            f"max_new_tokens ({max_new_tokens}) cannot exceed block_size ({block_size})",
        )
    old_config = json.loads(exp["config_json"])
    db_conn = await db.get_db()
    await db_conn.execute(
        "UPDATE experiments SET config_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(req.config), experiment_id),
    )
    await db_conn.commit()
    await db_conn.close()
    # Log config diff
    changed = {k: (old_config.get(k), v) for k, v in req.config.items() if old_config.get(k) != v}
    audit_log.info("Config updated: experiment_id=%d changed=%s", experiment_id, json.dumps(changed))
    return {"ok": True}


@router.patch("/{experiment_id}/notes")
async def update_notes(experiment_id: int, req: UpdateNotesRequest):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    db_conn = await db.get_db()
    await db_conn.execute(
        "UPDATE experiments SET notes_md = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (req.notes_md, experiment_id),
    )
    await db_conn.commit()
    await db_conn.close()
    audit_log.info("Notes updated: experiment_id=%d len=%d", experiment_id, len(req.notes_md))
    return {"ok": True}
