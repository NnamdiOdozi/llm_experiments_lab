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

# run_id -> in-flight background task for a remote run's provisioning
# (ensure_worker + mirror experiment + remote start). Lets Cancel/Stop
# actually interrupt provisioning instead of only being able to stop an
# already-running remote run. See _start_remote_run() and stop_training().
_provisioning_tasks: dict[int, asyncio.Task] = {}


class StartRunRequest(BaseModel):
    experiment_id: int
    device: str = "cpu"
    # Chosen per-request by the frontend, the same way device already is —
    # NOT settings.training_backend, which is only that dropdown's initial
    # suggestion. See docs/DESIGN_DECISIONS.md §11.
    backend: str = "local"


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


class UpdateRunNotesRequest(BaseModel):
    notes_md: str


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


async def _touch_worker_for_run(db_run: dict) -> None:
    """Reset the idle clock for whichever worker backs this run's device.

    Called after explicit user actions on a remote run (pause/resume/
    prompt) succeed — deliberately NOT wired into every _proxy() call,
    since passive status/metrics polling also goes through routes that use
    _proxy() and happens automatically on a timer regardless of whether
    anyone's actually engaged; touching on that too would make idle-timeout
    effectively never fire as long as a browser tab is left open. Confirmed
    live 2026-07-12: a user prompting a paused model for ~10 minutes (real,
    active engagement with the worker) saw an idle-timeout warning banner
    despite never being idle, because prompt_model() never touched
    last_activity_at at all — only worker acquisition and the manual
    "Continue session" heartbeat did. See docs/DESIGN_DECISIONS.md.
    """
    session_id = session_id_for(device_type_for(db_run["device"]))
    await db.touch_worker_session(session_id)


async def _start_remote_run(run_id: int, exp: dict, device: str) -> None:
    """Mirror the experiment onto the CPU/GPU endpoint and start training there.

    The endpoint is an ephemeral execution worker, not durable storage — the
    local training_runs row (updated below) stays the system of record. The
    frontend only ever deals with the local run_id; remote_run_id is an
    internal detail used to address the endpoint's own copy of the run.

    Runs as a backgrounded asyncio.Task (see start_training), not awaited
    directly inside the HTTP request — provisioning can take up to ~6
    minutes and the request must return immediately with the run_id so the
    frontend can poll status and Stop can cancel it mid-flight. Because
    there's no HTTP request context by the time this runs, failures update
    the run's DB status directly instead of raising HTTPException (nothing
    would receive it).
    """
    try:
        try:
            worker = await worker_manager.ensure_worker(device)
        except worker_manager.WorkerBusyError as exc:
            # Not a failure — a different request already has this device's
            # worker mid-provision. Leave the run's own status untouched (no
            # FAILED) so the user can just retry once the worker's ready,
            # instead of spawning a second endpoint for the same device.
            training_log.info("Worker busy, run not started — run_id=%d: %s", run_id, exc)
            return
        except worker_manager.WorkerProvisionError as exc:
            await db.update_training_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
            training_log.error("Worker provisioning failed — run_id=%d: %s", run_id, exc)
            return

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
            return

        await db.update_training_run(
            run_id, status=RunStatus.RUNNING, execution_backend="nebius_endpoint",
            remote_endpoint_id=worker["nebius_endpoint_id"], remote_run_id=remote_run_id,
        )
        await db.touch_worker_session(worker["session_id"])
        training_log.info(
            "Remote run started — run_id=%d remote_run_id=%d endpoint_id=%s device=%s",
            run_id, remote_run_id, worker["nebius_endpoint_id"], device,
        )
    except asyncio.CancelledError:
        # Cancelled via Stop while still provisioning (see stop_training) —
        # stop_training already sets CANCELLED itself, but set it here too
        # in case the cancellation lands between stop_training's DB write
        # and this task actually being scheduled to see it; both writes are
        # idempotent so there's no harm in doing it from both sides.
        await db.update_training_run(run_id, status=RunStatus.CANCELLED)
        training_log.info("Remote run provisioning cancelled — run_id=%d", run_id)
        raise


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
            # Set immediately from the user's actual choice, not left to the
            # 'local' schema default — _start_remote_run only used to set
            # this near the END of provisioning (after mirroring the
            # experiment remotely), which was invisible before Part F's
            # backgrounding change (the request blocked until it was set).
            # Now that /start returns immediately and the frontend polls
            # status right away, that gap became directly visible: a
            # serverless run showed "local" for its entire ~6min
            # provisioning window. See docs/DESIGN_DECISIONS.md.
            execution_backend=req.backend,
        )
        if req.backend == "nebius_endpoint":
            # Backgrounded, not awaited — provisioning can take up to ~6
            # minutes and the request must return the run_id immediately
            # (frontend polls status from there), same fire-and-forget
            # shape the local path below already uses. See _start_remote_run
            # and _provisioning_tasks for how Stop cancels this mid-flight.
            task = asyncio.create_task(_start_remote_run(run_id, exp, req.device))
            _provisioning_tasks[run_id] = task
            task.add_done_callback(lambda _t, rid=run_id: _provisioning_tasks.pop(rid, None))
        else:
            start_run(run_id, req.experiment_id, config, req.device)
        training_log.info(
            "START run_id=%d experiment_id=%d device=%s template=%s backend=%s",
            run_id, req.experiment_id, req.device, config.get("template", "transformer"),
            req.backend,
        )

        return {"run_id": run_id, "status": RunStatus.QUEUED}


def _is_remote(db_run: dict | None) -> bool:
    return db_run is not None and db_run.get("execution_backend") == "nebius_endpoint"


@router.get("/open")
async def list_open_runs():
    """Every non-terminal run across all experiments — feeds the Experiments
    page so a stuck run can be found and stopped even outside its own
    session's browser state.

    For remote runs, overlays live status/step from the remote endpoint —
    _start_remote_run never updates the local status column past QUEUED
    after handoff (only execution_backend/remote_endpoint_id/remote_run_id
    are set there), so the local row alone is permanently stale for any
    remote run once it starts actually training. Found live 2026-07-12: a
    successfully-running remote run showed QUEUED/step 0 forever in this
    list. db.list_open_runs()'s terminal-status filter also only sees the
    stale local status, so a remote run that's genuinely completed/failed
    would otherwise never drop out of this list either — filtered again
    here after the live overlay. Graceful per-run degradation: a proxy
    failure logs and keeps the stale local value rather than breaking the
    whole list. See docs/DESIGN_DECISIONS.md.
    """
    runs = await db.list_open_runs()
    for run in runs:
        if not _is_remote(run):
            continue
        try:
            live = await _proxy(run, "GET", "/api/training/{run_id}/status")
        except (httpx.HTTPError, HTTPException) as exc:
            training_log.warning(
                "Open Runs: live status fetch failed, showing stale local value — run_id=%d: %s",
                run["id"], exc,
            )
            continue
        run["status"] = live.get("status", run["status"])
        run["current_step"] = live.get("current_step", run["current_step"])
        run["total_steps"] = live.get("total_steps", run["total_steps"])
    return [r for r in runs if r["status"] not in TERMINAL_STATUSES]


@router.post("/{run_id}/pause")
async def pause_training(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            await _proxy(db_run, "POST", "/api/training/{run_id}/pause")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote pause failed: {exc}")
        await _touch_worker_for_run(db_run)
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
        await _touch_worker_for_run(db_run)
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
    # Check for an in-flight provisioning task first — execution_backend
    # isn't set on the run until _start_remote_run finishes successfully,
    # so _is_remote(db_run) below would be False the whole time it's still
    # provisioning, and stop_run() (the local-only path) would report "not
    # found" for it. Without this, there was no way to stop a remote run
    # during its up-to-~6min provisioning window at all.
    task = _provisioning_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        await db.update_training_run(run_id, status=RunStatus.CANCELLED)
        training_log.info("STOP (cancelled in-flight provisioning) run_id=%d", run_id)
        return {"run_id": run_id, "status": "stopping"}

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
        await _touch_worker_for_run(db_run)
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


@router.get("/{run_id}/notes")
async def get_run_notes(run_id: int):
    run = await db.get_training_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return {"notes_md": run.get("notes_md") or ""}


@router.patch("/{run_id}/notes")
async def update_run_notes(run_id: int, req: UpdateRunNotesRequest):
    run = await db.get_training_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    await db.update_training_run(run_id, notes_md=req.notes_md)
    training_log.info("Notes updated: run_id=%d len=%d", run_id, len(req.notes_md))
    return {"ok": True}


@router.get("/{run_id}/status")
async def run_status(run_id: int):
    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        if db_run.get("remote_run_id") is None:
            # _start_remote_run hasn't finished mirroring this run to the
            # endpoint yet — provisioning can take several minutes. Proxying
            # now would build a URL with a None remote_run_id and 404/502,
            # which the frontend can't tell apart from a genuine outage.
            # The local row is still legitimately QUEUED at this point, so
            # just serve that instead of failing. See docs/DESIGN_DECISIONS.md.
            local_status = await db.get_run_status_from_db(run_id)
            if local_status is not None:
                return local_status
        try:
            result = await _proxy(db_run, "GET", "/api/training/{run_id}/status")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote status fetch failed: {exc}")
        result["run_id"] = run_id  # never leak the endpoint's own remote run_id
        # The endpoint's own status.json thinks it's "local" from its own
        # perspective (it's a local subprocess to that container) — from the
        # controller's perspective this run is remote. Override, don't trust
        # whatever the proxied response says. See docs/DESIGN_DECISIONS.md §10.
        result["execution_backend"] = "nebius_endpoint"
        live_status = result.get("status")
        if live_status is not None and live_status != db_run["status"]:
            # Keep the local row in sync with whatever's actually happening
            # remotely — see §16/§17, the local status column otherwise
            # never advances past its creation value except via explicit
            # local pause/resume/prompt actions. Not special-cased to
            # "completed" — paused, cancelled, failed all matter equally.
            await db.update_training_run(
                run_id, status=live_status,
                current_step=result.get("current_step", db_run["current_step"]),
                total_steps=result.get("total_steps", db_run["total_steps"]),
            )
            if live_status in TERMINAL_STATUSES and db_run["status"] not in TERMINAL_STATUSES:
                # The run just finished (completed/failed/cancelled) on its
                # own, without an explicit local stop/pause action to have
                # already touched the clock — that's still a legitimate
                # "something real just happened here" signal.
                await _touch_worker_for_run(db_run)
        return result
    # Try in-memory first (live run), then fall back to DB (after restart)
    status = get_run_status(run_id)
    if status is not None:
        return status
    db_status = await db.get_run_status_from_db(run_id)
    if db_status is not None:
        return db_status
    raise HTTPException(404, "Run not found")


def read_metrics_from_disk(run_id: int) -> list[dict]:
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
        if db_run.get("remote_run_id") is None:
            # Same reasoning as run_status() — nothing to proxy to yet while
            # _start_remote_run is still mirroring the run, and there
            # genuinely are no metrics for a run that hasn't started. See
            # docs/DESIGN_DECISIONS.md. Without this, the frontend's poll
            # loop calls this right after a successful status call and its
            # failure alone was enough to trip the disconnect banner, even
            # though status was already correctly reporting "queued".
            return []
        try:
            metrics = await _proxy(db_run, "GET", "/api/training/{run_id}/metrics")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote metrics fetch failed: {exc}")
        if metrics:
            # Mirrors what train_worker.py::write_metric() does for local
            # runs — without this, train_loss_history on the local row stays
            # '[]' forever for a remote run (nothing else ever writes it),
            # so the chatbot's loss-trend snapshot (context.py) always sees
            # no data no matter how far training has actually progressed.
            # Piggybacks on this route rather than adding a separate proxy
            # call, since the frontend already polls it every ~2s anyway.
            # See docs/DESIGN_DECISIONS.md.
            await db.update_training_run(
                run_id,
                train_loss_history=json.dumps([m for m in metrics if "train_loss" in m]),
                val_loss_history=json.dumps([m for m in metrics if "val_loss" in m]),
            )
        return metrics
    # Always read from disk (worker writes metrics.jsonl)
    disk_metrics = read_metrics_from_disk(run_id)
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
            current_metrics = read_metrics_from_disk(run_id)
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
