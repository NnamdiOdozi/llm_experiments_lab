"""Experiment CRUD endpoints."""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import db
from backend.logging_config import audit_log
from config.presets import PRESETS
from backend.training.tokenizers.loader import load_tokenizer
from data.tokenizers.manifest import load_manifest

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def normalize_config(config: dict, template: str = None) -> dict:
    """Ensure config has a valid 'data' block with derived vocab_size.

    Args:
        config: Experiment config dict.
        template: Optional template key to infer dataset (e.g. 'rnn' → dinos).

    Returns:
        Config dict with a 'data' block guaranteed to exist and be valid.
        Absent config["data"] becomes char/tiny_shakespeare for transformer/moe,
        or char/dinos for rnn. vocab_size is derived from the tokenizer.
    """
    template = template or config.get("template", "transformer")

    if "data" not in config:
        if template == "rnn":
            config["data"] = {
                "dataset": "dinos",
                "tokenizer": "char",
                "tokenizer_artifact": None,
                "vocab_size": 29,
            }
        else:  # transformer, moe, or unknown → default to tiny_shakespeare
            config["data"] = {
                "dataset": "tiny_shakespeare",
                "tokenizer": "char",
                "tokenizer_artifact": None,
                "vocab_size": 65,
            }
    else:
        # RNN doesn't use tokenizers; don't derive vocab_size
        if template != "rnn":
            # Recompute vocab_size from the tokenizer for transformer/moe
            try:
                tokenizer = load_tokenizer(config["data"])
                config["data"]["vocab_size"] = tokenizer.vocab_size

                # Also fetch tokenizer metadata from manifest for checkpoint storage
                if config["data"]["tokenizer"] != "char":
                    manifest = load_manifest()
                    for tok_entry in manifest.get("tokenizers", []):
                        if tok_entry.get("id") == config["data"]["tokenizer"]:
                            config["data"]["tokenizer_version"] = tok_entry.get("version", "unknown")
                            config["data"]["tokenizer_hash"] = tok_entry.get("sha256", "unknown")
                            break
            except Exception as e:
                # If tokenizer loading fails, use what's in config (for old/invalid configs)
                pass

    return config


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
    config = normalize_config(req.config, template=req.config.get("template"))
    exp_id = await db.create_experiment(req.name, config, req.preset_key)
    audit_log.info("Experiment created: id=%d name='%s' preset=%s", exp_id, req.name, req.preset_key)
    return {"id": exp_id, "name": req.name}


@router.post("/from-preset/{preset_key}")
async def create_from_preset(preset_key: str):
    if preset_key not in PRESETS:
        raise HTTPException(404, f"Preset '{preset_key}' not found")
    preset = PRESETS[preset_key]
    preset = normalize_config(preset, template=preset.get("template"))
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
    exp["config"] = normalize_config(exp["config"], template=exp["config"].get("template"))
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
