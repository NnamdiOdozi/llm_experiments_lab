"""Training control endpoints + WebSocket for metrics streaming."""

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend import db
from backend.logging_config import training_log, prompt_log
from backend.training.runner import (
    active_runs,
    start_run,
    pause_run,
    resume_run,
    stop_run,
    prompt_paused_model,
    get_run_status,
)

router = APIRouter(prefix="/api/training", tags=["training"])


class StartRunRequest(BaseModel):
    experiment_id: int
    device: str = "cpu"


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@router.post("/start")
async def start_training(req: StartRunRequest):
    exp = await db.get_experiment(req.experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")

    config = json.loads(exp["config_json"])
    run_id = await db.create_training_run(req.experiment_id, req.device)
    start_run(run_id, req.experiment_id, config, req.device)
    training_log.info(
        "START run_id=%d experiment_id=%d device=%s template=%s",
        run_id, req.experiment_id, req.device, config.get("template", "transformer"),
    )

    return {"run_id": run_id, "status": "queued"}


@router.post("/{run_id}/pause")
async def pause_training(run_id: int):
    if not pause_run(run_id):
        raise HTTPException(400, "Run not found or not running")
    training_log.info("PAUSE requested run_id=%d", run_id)
    return {"run_id": run_id, "status": "pausing"}


@router.post("/{run_id}/resume")
async def resume_training(run_id: int):
    if not resume_run(run_id):
        raise HTTPException(400, "Run not found or not paused")
    training_log.info("RESUME run_id=%d", run_id)
    return {"run_id": run_id, "status": "resuming"}


@router.post("/{run_id}/stop")
async def stop_training(run_id: int):
    if not stop_run(run_id):
        raise HTTPException(400, "Run not found")
    training_log.info("STOP run_id=%d", run_id)
    return {"run_id": run_id, "status": "stopping"}


@router.post("/{run_id}/prompt")
async def prompt_model(run_id: int, req: PromptRequest):
    result = prompt_paused_model(run_id, req.prompt, req.max_new_tokens)
    if result is None:
        raise HTTPException(400, "Run not paused or model not available")
    prompt_log.info(
        "run_id=%d prompt='%s' max_tokens=%d output_len=%d",
        run_id, req.prompt[:50], req.max_new_tokens, len(result),
    )
    return {"run_id": run_id, "prompt": req.prompt, "output": result}


@router.get("/{run_id}/status")
async def run_status(run_id: int):
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(404, "Run not found")
    return status


@router.get("/{run_id}/metrics")
async def get_metrics(run_id: int):
    run = active_runs.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run.metrics


@router.websocket("/{run_id}/ws")
async def metrics_websocket(websocket: WebSocket, run_id: int):
    """Stream metrics to the browser as they arrive."""
    await websocket.accept()
    last_sent = 0

    try:
        while True:
            run = active_runs.get(run_id)
            if run is None:
                await websocket.send_json({"type": "error", "message": "Run not found"})
                break

            # Send new metrics
            current_metrics = run.metrics
            if len(current_metrics) > last_sent:
                for metric in current_metrics[last_sent:]:
                    await websocket.send_json({"type": "metric", "data": metric})
                last_sent = len(current_metrics)

            # Send status updates
            await websocket.send_json({
                "type": "status",
                "status": run.status.value,
                "current_step": run.current_step,
                "total_steps": run.config["training"]["max_iters"],
            })

            # Stop streaming if run is done
            if run.status.value in ("completed", "failed"):
                await websocket.send_json({"type": "done", "status": run.status.value})
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
