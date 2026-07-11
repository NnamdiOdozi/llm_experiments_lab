"""Training control endpoints + WebSocket for metrics streaming."""

import asyncio
import json

import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend import db
from backend.logging_config import training_log, prompt_log
from backend.nebius import worker_manager
from backend.training.runner import (
    active_runs,
    start_run,
    pause_run,
    resume_run,
    stop_run,
    prompt_paused_model,
    get_run_status,
)
from backend.training.status import RunStatus, TERMINAL_STATUSES
from backend.training.worker_status import device_type_for, session_id_for
from config.settings import settings

router = APIRouter(prefix="/api/training", tags=["training"])

# Serialize start requests to prevent race conditions
_start_lock = asyncio.Lock()


class StartRunRequest(BaseModel):
    experiment_id: int
    device: str = "cpu"


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


def _count_active_runs(device_filter: str | None = None) -> int:
    """Count runs with live worker processes."""
    count = 0
    for r in active_runs.values():
        if r.process.poll() is not None:
            continue  # process finished
        if device_filter is None or r.device.startswith(device_filter):
            count += 1
    return count


async def _remote_endpoint_url(db_run: dict) -> str | None:
    """The endpoint currently backing this run's device, if it's still known."""
    session_id = session_id_for(device_type_for(db_run["device"]))
    worker = await db.get_worker_session(session_id)
    return worker["endpoint_url"] if worker else None


async def _proxy(db_run: dict, method: str, path: str, json_body: dict | None = None) -> dict:
    """Forward a training-control call to the remote endpoint, using its own
    remote_run_id — the frontend never sees that id, only the local run_id.
    """
    endpoint_url = await _remote_endpoint_url(db_run)
    if endpoint_url is None:
        raise HTTPException(502, "Remote worker endpoint not available")
    remote_path = path.format(run_id=db_run["remote_run_id"])
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, f"{endpoint_url}{remote_path}", json=json_body)
    resp.raise_for_status()
    return resp.json()


async def _start_remote_run(run_id: int, exp: dict, device: str) -> None:
    """Mirror the experiment onto the CPU/GPU endpoint and start training there.

    The endpoint is an ephemeral execution worker, not durable storage — the
    local training_runs row (updated below) stays the system of record. The
    frontend only ever deals with the local run_id; remote_run_id is an
    internal detail used to address the endpoint's own copy of the run.
    """
    try:
        worker = await worker_manager.ensure_worker(device)
    except worker_manager.WorkerProvisionError as exc:
        await db.update_training_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
        training_log.error("Worker provisioning failed — run_id=%d: %s", run_id, exc)
        raise HTTPException(502, f"Remote worker unavailable: {exc}")

    endpoint_url = worker["endpoint_url"]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            exp_resp = await client.request(
                "POST", f"{endpoint_url}/api/experiments",
                json={"name": exp["name"], "config": json.loads(exp["config_json"]), "preset_key": exp["preset_key"]},
            )
            exp_resp.raise_for_status()
            remote_experiment_id = exp_resp.json()["id"]

            run_resp = await client.request(
                "POST", f"{endpoint_url}/api/training/start",
                json={"experiment_id": remote_experiment_id, "device": device},
            )
            run_resp.raise_for_status()
            remote_run_id = run_resp.json()["run_id"]
    except httpx.HTTPError as exc:
        await db.update_training_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
        training_log.error("Remote training start failed — run_id=%d: %s", run_id, exc)
        raise HTTPException(502, f"Remote training start failed: {exc}")

    await db.update_training_run(
        run_id, execution_backend="nebius_endpoint",
        remote_endpoint_id=worker["nebius_endpoint_id"], remote_run_id=remote_run_id,
    )
    await db.touch_worker_session(worker["session_id"])
    training_log.info(
        "Remote run started — run_id=%d remote_run_id=%d endpoint_id=%s device=%s",
        run_id, remote_run_id, worker["nebius_endpoint_id"], device,
    )


@router.post("/start")
async def start_training(req: StartRunRequest):
    async with _start_lock:
        # Enforce concurrency limits — check both in-memory and DB
        active_total = max(
            _count_active_runs(),
            await db.count_active_runs_in_db(),
        )
        if active_total >= settings.max_concurrent_runs:
            raise HTTPException(
                429, f"Max {settings.max_concurrent_runs} concurrent runs. Stop a run first."
            )
        if req.device.startswith("cuda"):
            gpu_count = max(
                _count_active_runs("cuda"),
                await db.count_active_runs_in_db("cuda"),
            )
            if gpu_count >= settings.max_concurrent_gpu_runs:
                raise HTTPException(
                    429, f"Max {settings.max_concurrent_gpu_runs} GPU run(s). Stop the GPU run first."
                )

        exp = await db.get_experiment(req.experiment_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")

        config = json.loads(exp["config_json"])
        run_id = await db.create_training_run(
            req.experiment_id, req.device,
            config_snapshot=json.dumps(config),
            template_key=config.get("template", "transformer"),
        )
        if settings.training_backend == "nebius_endpoint":
            await _start_remote_run(run_id, exp, req.device)
        else:
            start_run(run_id, req.experiment_id, config, req.device)
        training_log.info(
            "START run_id=%d experiment_id=%d device=%s template=%s backend=%s",
            run_id, req.experiment_id, req.device, config.get("template", "transformer"),
            settings.training_backend,
        )

        return {"run_id": run_id, "status": RunStatus.QUEUED}


def _is_remote(db_run: dict | None) -> bool:
    return db_run is not None and db_run.get("execution_backend") == "nebius_endpoint"


@router.get("/open")
async def list_open_runs():
    """Every non-terminal run across all experiments — feeds the Experiments
    page so a stuck run can be found and stopped even outside its own
    session's browser state.
    """
    return await db.list_open_runs()


@router.post("/{run_id}/pause")
async def pause_training(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            await _proxy(db_run, "POST", "/api/training/{run_id}/pause")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote pause failed: {exc}")
        training_log.info("PAUSE requested (remote) run_id=%d", run_id)
        return {"run_id": run_id, "status": "pausing"}
    if not pause_run(run_id):
        raise HTTPException(400, "Run not found or not running")
    training_log.info("PAUSE requested run_id=%d", run_id)
    return {"run_id": run_id, "status": "pausing"}


@router.post("/{run_id}/resume")
async def resume_training(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            # NOTE: does not push config edits made while paused to the remote
            # mirrored experiment — known gap, local resume already does this
            # (see updated_config below), remote resume doesn't yet.
            await _proxy(db_run, "POST", "/api/training/{run_id}/resume")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote resume failed: {exc}")
        training_log.info("RESUME (remote) run_id=%d", run_id)
        return {"run_id": run_id, "status": RunStatus.RESUMING}
    # Fetch latest config from DB so edits made while paused
    # (e.g. max_iters, eval_interval, inference params) take effect.
    updated_config = None
    if db_run:
        exp = await db.get_experiment(db_run["experiment_id"])
        if exp:
            updated_config = json.loads(exp["config_json"])
    if not resume_run(run_id, updated_config):
        raise HTTPException(400, "Run not found or not paused")
    training_log.info("RESUME run_id=%d config_refreshed=%s", run_id, updated_config is not None)
    return {"run_id": run_id, "status": RunStatus.RESUMING}


@router.post("/{run_id}/stop")
async def stop_training(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            await _proxy(db_run, "POST", "/api/training/{run_id}/stop")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote stop failed: {exc}")
        training_log.info("STOP (remote) run_id=%d", run_id)
        return {"run_id": run_id, "status": "stopping"}
    if not stop_run(run_id):
        raise HTTPException(400, "Run not found")
    training_log.info("STOP run_id=%d", run_id)
    return {"run_id": run_id, "status": "stopping"}


@router.post("/{run_id}/prompt")
async def prompt_model(run_id: int, req: PromptRequest):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(
                db_run, "POST", "/api/training/{run_id}/prompt",
                {"prompt": req.prompt, "max_new_tokens": req.max_new_tokens},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote prompt failed: {exc}")
        output = result.get("output")
        prompt_log.info(
            "run_id=%d (remote) payload=%s", run_id,
            json.dumps({"prompt": req.prompt, "output": output}),
        )
        return {"run_id": run_id, "prompt": req.prompt, "output": output}
    result = prompt_paused_model(run_id, req.prompt, req.max_new_tokens)
    if result is None:
        raise HTTPException(400, "Run not paused or model not available")
    # Full prompt+output logged as one JSON payload — the chatbot's context
    # builder (backend/chatbot/context.py::_get_prompt_history) parses these
    # lines back out so the Lab Assistant can see pause-and-prompt exchanges.
    status = get_run_status(run_id) or {}
    prompt_log.info(
        "run_id=%d payload=%s",
        run_id,
        json.dumps({"step": status.get("current_step"), "prompt": req.prompt, "output": result}),
    )
    return {"run_id": run_id, "prompt": req.prompt, "output": result}


@router.get("/{run_id}/status")
async def run_status(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(db_run, "GET", "/api/training/{run_id}/status")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote status fetch failed: {exc}")
        result["run_id"] = run_id  # never leak the endpoint's own remote run_id
        return result
    # Try in-memory first (live run), then fall back to DB (after restart)
    status = get_run_status(run_id)
    if status is not None:
        return status
    db_status = await db.get_run_status_from_db(run_id)
    if db_status is not None:
        return db_status
    raise HTTPException(404, "Run not found")


def _read_metrics_from_disk(run_id: int) -> list[dict]:
    """Read metrics.jsonl from disk for runs no longer in memory."""
    metrics_file = settings.data_dir / "runs" / str(run_id) / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupt/partial lines from crashed runs
    return metrics


@router.get("/{run_id}/metrics")
async def get_metrics(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            return await _proxy(db_run, "GET", "/api/training/{run_id}/metrics")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote metrics fetch failed: {exc}")
    # Always read from disk (worker writes metrics.jsonl)
    disk_metrics = _read_metrics_from_disk(run_id)
    if disk_metrics:
        return disk_metrics
    # Check if run exists (active or in DB)
    if run_id in active_runs:
        return []
    if db_run is None:
        raise HTTPException(404, "Run not found")
    return []


@router.websocket("/{run_id}/ws")
async def metrics_websocket(websocket: WebSocket, run_id: int):
    """Stream metrics to the browser as they arrive."""
    from backend.training import artifacts

    await websocket.accept()
    last_sent = 0
    db_run = await db.get_training_run(run_id)

    try:
        if _is_remote(db_run):
            # Endpoint has no push channel back to us, so poll its REST routes
            # on the same cadence the local branch below polls disk.
            while True:
                try:
                    status = await _proxy(db_run, "GET", "/api/training/{run_id}/status")
                    current_metrics = await _proxy(db_run, "GET", "/api/training/{run_id}/metrics")
                except httpx.HTTPError:
                    await websocket.send_json({"type": "error", "message": "Remote worker unreachable"})
                    break

                if isinstance(current_metrics, list) and len(current_metrics) > last_sent:
                    for metric in current_metrics[last_sent:]:
                        await websocket.send_json({"type": "metric", "data": metric})
                    last_sent = len(current_metrics)

                await websocket.send_json({
                    "type": "status",
                    "status": status.get("status"),
                    "current_step": status.get("current_step", 0),
                    "total_steps": status.get("total_steps", 0),
                })

                if status.get("status") in TERMINAL_STATUSES:
                    await websocket.send_json({"type": "done", "status": status.get("status")})
                    break

                await asyncio.sleep(2)
            return

        while True:
            status = artifacts.read_status(run_id)
            if status is None:
                await websocket.send_json({"type": "error", "message": "Run not found"})
                break

            # Send new metrics from disk
            current_metrics = _read_metrics_from_disk(run_id)
            if len(current_metrics) > last_sent:
                for metric in current_metrics[last_sent:]:
                    await websocket.send_json({"type": "metric", "data": metric})
                last_sent = len(current_metrics)

            # Send status updates
            await websocket.send_json({
                "type": "status",
                "status": status["status"],
                "current_step": status.get("current_step", 0),
                "total_steps": status.get("total_steps", 0),
            })

            # Stop streaming if run is done
            if status["status"] in TERMINAL_STATUSES:
                await websocket.send_json({"type": "done", "status": status["status"]})
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
