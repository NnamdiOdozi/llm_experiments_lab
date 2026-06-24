"""Experiment CRUD endpoints."""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import db
from config.presets import PRESETS

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class CreateExperimentRequest(BaseModel):
    name: str
    config: dict
    preset_key: str | None = None


class UpdateNotesRequest(BaseModel):
    notes_md: str


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
    return {"id": exp_id, "name": req.name}


@router.post("/from-preset/{preset_key}")
async def create_from_preset(preset_key: str):
    if preset_key not in PRESETS:
        raise HTTPException(404, f"Preset '{preset_key}' not found")
    preset = PRESETS[preset_key]
    exp_id = await db.create_experiment(preset["name"], preset, preset_key)
    return {"id": exp_id, "name": preset["name"], "config": preset}


@router.get("/presets")
async def list_presets():
    return PRESETS


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    exp["config"] = json.loads(exp["config_json"])
    del exp["config_json"]
    return exp


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
    return {"ok": True}
