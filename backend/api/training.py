"""Training control endpoints + WebSocket for metrics streaming."""

import asyncio
import itertools
import json
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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


class DiagnosticsStartRequest(BaseModel):
    prompt: str
    top_k: int = 5
    max_prompt_tokens: int = 32


class DiagnosticsStepRequest(BaseModel):
    """POST body for step endpoint. Phase 1: always captures shapes + top-k.
    Phase 2: optional attention_layer/head triggers an explicit attention
    capture for that layer/head; omit both to skip it (cheap default).
    Phase 4: qkv_detail (requires attention_layer/head) adds Q/K/V vectors
    for the last token position, one head only."""
    attention_layer: int | None = None
    attention_head: int | None = None
    qkv_detail: bool = False
    # Shifts the attention heatmap/qkv_detail window earlier in the
    # sequence — 0 (default) shows the most recent DIAGNOSTIC_POSITION_WINDOW
    # positions, positive N shifts back N positions. Lets the Inspector's
    # heatmap stepper browse history instead of only ever showing the tail
    # (real user report, 2026-07-13: heatmap "gets very busy very quickly").
    # See docs/DESIGN_DECISIONS.md.
    attention_window_offset: int = 0


class DiagnosticsGenerateRequest(BaseModel):
    """POST body for the Phase 3 streaming /generate endpoint."""
    max_new_tokens: int = 50
    attention_layer: int | None = None
    attention_head: int | None = None
    qkv_detail: bool = False


def _count_active_runs(device_filter: str | None = None) -> int:
    """Count runs with live worker processes.

    active_runs (backend/training/runner.py) only ever holds LOCAL runs —
    _start_remote_run() never calls start_run(), so a serverless run never
    gets an entry here. Meaningful only for the local-backend concurrency
    check; see start_training()."""
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
        # Enforce concurrency limits — separate per device x execution
        # backend (a laptop's local capacity is unrelated to how many
        # concurrent serverless endpoint sessions are allowed). Check both
        # in-memory (local runs only) and DB (survives API restarts, covers
        # both backends) and take the max. See docs/DESIGN_DECISIONS.md.
        is_gpu = req.device.startswith("cuda")
        is_serverless = req.backend == "nebius_endpoint"
        device_filter = "cuda" if is_gpu else "cpu"
        backend_filter = "nebius_endpoint" if is_serverless else "local"
        if is_serverless:
            limit = (
                settings.max_concurrent_serverless_gpu_runs
                if is_gpu else settings.max_concurrent_serverless_cpu_runs
            )
        else:
            limit = (
                settings.max_concurrent_local_gpu_runs
                if is_gpu else settings.max_concurrent_local_cpu_runs
            )

        active_count = await db.count_active_runs_in_db(device_filter, backend_filter)
        if not is_serverless:
            active_count = max(active_count, _count_active_runs(device_filter))

        if active_count >= limit:
            kind = "GPU" if is_gpu else "CPU"
            where = "serverless" if is_serverless else "local"
            raise HTTPException(
                429, f"Max {limit} concurrent {where} {kind} run(s). Stop a run first."
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
    # Check local status before ever touching the network. Previously
    # pause always proxied straight through for remote runs — if the run
    # had already reached completed/failed/cancelled (e.g. the frontend's
    # step counter hadn't caught up with the last poll yet), the remote's
    # own pause_run() correctly returned 400, but _proxy()'s
    # raise_for_status() turned that into an httpx error which got
    # collapsed into a generic, unhelpful 502 here. Real bug report,
    # 2026-07-13. See docs/DESIGN_DECISIONS.md.
    if db_run["status"] in TERMINAL_STATUSES:
        raise HTTPException(400, f"Cannot pause — run is already {db_run['status']}")
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
        raise HTTPException(400, "Run must be paused or completed, with a saved checkpoint, to prompt")
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
    """Stream metrics to the browser as they arrive.

    Messages wrapped in the standard event envelope (schema_version,
    event_id, timestamp, local_run_id, remote_run_id, type, payload) per
    docs/Diagnostic_Contract.md / docs/Trainer_to_Frontend_Metrics.md — the
    transport itself (this WS, polling disk/remote every 2s) is unchanged,
    only the message shape gained a stable envelope around existing content.
    """
    from backend.training import artifacts

    await websocket.accept()
    last_sent = 0
    db_run = await db.get_training_run(run_id)
    remote_run_id = db_run.get("remote_run_id") if db_run else None
    event_ids = itertools.count(1)

    async def send_event(event_type: str, payload: dict) -> None:
        await websocket.send_json({
            "schema_version": 1,
            "event_id": next(event_ids),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_run_id": run_id,
            "remote_run_id": remote_run_id,
            "type": event_type,
            "payload": payload,
        })

    try:
        if _is_remote(db_run):
            # Endpoint has no push channel back to us, so poll its REST routes
            # on the same cadence the local branch below polls disk.
            while True:
                try:
                    status = await _proxy(db_run, "GET", "/api/training/{run_id}/status")
                    current_metrics = await _proxy(db_run, "GET", "/api/training/{run_id}/metrics")
                except httpx.HTTPError:
                    await send_event("error", {"message": "Remote worker unreachable"})
                    break

                if isinstance(current_metrics, list) and len(current_metrics) > last_sent:
                    for metric in current_metrics[last_sent:]:
                        await send_event("metric", {"data": metric})
                    last_sent = len(current_metrics)

                await send_event("status", {
                    "status": status.get("status"),
                    "current_step": status.get("current_step", 0),
                    "total_steps": status.get("total_steps", 0),
                })

                if status.get("status") in TERMINAL_STATUSES:
                    await send_event("done", {"status": status.get("status")})
                    break

                await asyncio.sleep(2)
            return

        while True:
            status = artifacts.read_status(run_id)
            if status is None:
                await send_event("error", {"message": "Run not found"})
                break

            # Send new metrics from disk
            current_metrics = read_metrics_from_disk(run_id)
            if len(current_metrics) > last_sent:
                for metric in current_metrics[last_sent:]:
                    await send_event("metric", {"data": metric})
                last_sent = len(current_metrics)

            # Send status updates
            await send_event("status", {
                "status": status["status"],
                "current_step": status.get("current_step", 0),
                "total_steps": status.get("total_steps", 0),
            })

            # Stop streaming if run is done
            if status["status"] in TERMINAL_STATUSES:
                await send_event("done", {"status": status["status"]})
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass


@router.get("/{run_id}/architecture")
async def get_architecture_manifest(run_id: int):
    """Return static architecture manifest derived from config.

    For local runs, derive from run_dir/config.json. For remote runs,
    proxy to the endpoint. Node IDs and Phase 1 scope per
    docs/Diagnostic_Contract.md.
    """
    from backend.training import artifacts
    from backend.training.templates import TEMPLATE_REGISTRY

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(db_run, "GET", "/api/training/{run_id}/architecture")
            result["local_run_id"] = run_id  # Override with local run_id
            return result
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote architecture fetch failed: {exc}")

    # Load config from disk
    rd = artifacts.run_dir(run_id)
    config_path = rd / "config.json"
    if not config_path.exists():
        raise HTTPException(404, "Run config not found")

    config = json.loads(config_path.read_text())
    template_key = config.get("template", "transformer")
    model_cfg = config.get("model", {})

    # Count parameters using template's model
    total_params = 0
    trainable_params = 0
    try:
        model = TEMPLATE_REGISTRY[template_key]["build_model"](config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        training_log.warning("Could not count params for run %d (may not have a saved checkpoint yet): %s", run_id, e)

    # RNN (CharRNN) has a completely different shape from transformer/MoE —
    # one-hot input, a single stacked nn.LSTM (n_layers is internal to one
    # module, not separate Block instances), dropout, then a linear head.
    # No attention, no residual blocks, no separate per-layer boxes — a
    # deliberately simpler diagram per user request (2026-07-13: "easier
    # than the transformer/MoE one, come up with something sensible").
    if template_key == "rnn":
        n_hidden = model_cfg.get("n_hidden", 256)
        n_layers = model_cfg.get("n_layers", 2)
        rnn_nodes = [
            {
                "id": "one_hot",
                "kind": "embedding",
                "label": "One-Hot Encoding",
                "config": {"vocab_size": model_cfg.get("vocab_size")},
                "static_shapes": [
                    {"name": "input", "dims": ["batch", "sequence"]},
                    {"name": "output", "dims": ["batch", "sequence", "vocab_size"]},
                ],
                "math_key": "one_hot",
            },
            {
                "id": "lstm",
                "kind": "rnn",
                "label": f"LSTM ({n_layers} layer{'s' if n_layers != 1 else ''}, stacked)",
                "config": {"n_hidden": n_hidden, "n_layers": n_layers, "dropout": model_cfg.get("dropout", 0.5)},
                "static_shapes": [
                    {"name": "input", "dims": ["batch", "sequence", "vocab_size"]},
                    {"name": "output", "dims": ["batch", "sequence", "n_hidden"]},
                ],
                "math_key": "lstm_cell",
            },
            {
                "id": "dropout",
                "kind": "dropout",
                "label": "Dropout",
                "config": {"dropout": model_cfg.get("dropout", 0.5)},
            },
            {
                "id": "lm_head",
                "kind": "lm_head",
                "label": "Linear (FC)",
                "config": {"vocab_size": model_cfg.get("vocab_size")},
                "static_shapes": [
                    {"name": "input", "dims": ["batch * sequence", "n_hidden"]},
                    {"name": "output", "dims": ["batch * sequence", "vocab_size"]},
                ],
            },
        ]
        return {
            "schema_version": 1,
            "local_run_id": run_id,
            "template": template_key,
            "param_count": total_params,
            "trainable_param_count": trainable_params,
            "nodes": rnn_nodes,
        }

    # Build nodes list
    nodes = []

    # Embedding node
    nodes.append({
        "id": "embedding",
        "kind": "embedding",
        "label": "Token + Positional Embedding",
        "config": {
            "vocab_size": model_cfg.get("vocab_size"),
            "n_embd": model_cfg.get("n_embd"),
            "pos_encoding": model_cfg.get("pos_encoding", "learned"),
        },
        "static_shapes": [
            {"name": "input", "dims": ["batch", "sequence"]},
            {"name": "output", "dims": ["batch", "sequence", "n_embd"]},
        ],
        "math_key": "embedding_lookup",
    })

    # Transformer/MoE blocks
    n_layer = model_cfg.get("n_layer", 4)
    n_head = model_cfg.get("n_head", 6)
    head_dim = model_cfg.get("n_embd", 192) // n_head
    dropout = model_cfg.get("dropout", 0.1)
    activation = model_cfg.get("activation", "gelu")

    block_children = [
        {"id": "block.{i}.ln1", "kind": "layernorm", "label": "LayerNorm (pre-attention)"},
        {"id": "block.{i}.attention", "kind": "attention", "label": "Causal Self-Attention",
         "math_key": "scaled_dot_product_attention"},
        {"id": "block.{i}.ln2", "kind": "layernorm", "label": "LayerNorm (pre-MLP/MoE)"},
    ]

    # MLP or MoE, depending on template — per user feedback (2026-07-13), a
    # MoE layer has multiple experts, not one FFN, so it gets a distinct
    # node kind/config here, not a relabeled "mlp" node.
    if template_key == "moe":
        moe_config = {
            "num_experts": model_cfg.get("num_experts", 8),
            "top_k": model_cfg.get("top_k", 2),
            "capacity_factor": model_cfg.get("capacity_factor", 1.25),
        }
        block_children.append({
            "id": "block.{i}.moe",
            "kind": "moe",
            "label": "Mixture-of-Experts",
            "config": moe_config,
        })
        block_group_label = "Transformer Block (MoE)"
    else:
        block_children.append({
            "id": "block.{i}.mlp",
            "kind": "mlp",
            "label": "Feed-Forward (dense)",
            "math_key": "mlp_gelu",
        })
        block_group_label = "Transformer Block"

    nodes.append({
        "id": "block",
        "kind": "transformer_block_group",
        "label": block_group_label,
        "repeat_count": n_layer,
        "config": {
            "n_head": n_head,
            "head_dim": head_dim,
            "dropout": dropout,
            "activation": activation,
        },
        "children": block_children,
    })

    # Final norm
    nodes.append({
        "id": "final_norm",
        "kind": "layernorm",
        "label": "Final LayerNorm",
        "config": {},
    })

    # LM head
    nodes.append({
        "id": "lm_head",
        "kind": "lm_head",
        "label": "LM Head",
        "config": {"vocab_size": model_cfg.get("vocab_size")},
        "static_shapes": [
            {"name": "input", "dims": ["batch", "sequence", "n_embd"]},
            {"name": "output", "dims": ["batch", "sequence", "vocab_size"]},
        ],
    })

    return {
        "schema_version": 1,
        "local_run_id": run_id,
        "template": template_key,
        "param_count": total_params,
        "trainable_param_count": trainable_params,
        "nodes": nodes,
    }


@router.get("/{run_id}/architecture/embedding-table")
async def get_embedding_table(run_id: int):
    """Return the trained token embedding matrix (vocab_size x n_embd).

    Direct user request 2026-07-15 — the Inspector's embedding node only
    ever showed per-position runtime vectors for the current prompt, not
    the underlying learned table itself. Requires a saved checkpoint (the
    static /architecture route builds a fresh untrained model just to
    count params — this one needs real trained weights). transformer/moe
    only: RNN's CharRNN has no token_emb (uses one-hot input, see
    get_architecture_manifest's rnn branch), so there's no embedding
    matrix to show. See docs/DESIGN_DECISIONS.md.
    """
    from backend.training import artifacts
    from backend.training.templates import TEMPLATE_REGISTRY

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(db_run, "GET", "/api/training/{run_id}/architecture/embedding-table")
            return result
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote embedding table fetch failed: {exc}")

    rd = artifacts.run_dir(run_id)
    config_path = rd / "config.json"
    if not config_path.exists():
        raise HTTPException(404, "Run config not found")
    config = json.loads(config_path.read_text())
    template_key = config.get("template", "transformer")
    if template_key not in ("transformer", "moe"):
        raise HTTPException(400, f"No embedding table for template: {template_key}")

    cp_path = artifacts.checkpoint_path(run_id)
    if not cp_path.exists():
        raise HTTPException(400, "No checkpoint available for this run")

    import torch
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    model_config = cp.get("config", config)
    model = TEMPLATE_REGISTRY[template_key]["build_model"](model_config)
    model.load_state_dict(cp["model_state"])
    model.eval()

    from backend.training.templates.transformer.data import load_tiny_shakespeare, CharDataset
    text = load_tiny_shakespeare()
    tokenizer = CharDataset(text, model_config["model"]["block_size"], 1)

    weight = model.token_emb.weight.detach().cpu()
    vocab_size, n_embd = weight.shape

    # Position embedding table is only a real learned parameter under
    # pos_encoding="learned" (nn.Embedding(block_size, n_embd)) — RoPE
    # computes rotary embeddings on the fly, no table exists to show.
    # hasattr is the ground truth here (checked against the actual loaded
    # model), not the config string, so this can't drift out of sync with
    # the model's real structure. Direct user follow-up, 2026-07-13: "I
    # can't see the position embedding table... I think they should both
    # be on that tab." See docs/DESIGN_DECISIONS.md.
    position_embedding = None
    block_size = None
    if hasattr(model, "pos_emb"):
        pos_weight = model.pos_emb.weight.detach().cpu()
        block_size = pos_weight.shape[0]
        position_embedding = pos_weight.tolist()

    return {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "tokens": [tokenizer.decode([i]) for i in range(vocab_size)],
        "embedding": weight.tolist(),
        "block_size": block_size,
        "position_embedding": position_embedding,
    }


@router.post("/{run_id}/diagnostics/start")
async def diagnostics_start(run_id: int, req: DiagnosticsStartRequest):
    """Initialize a diagnostic session: load model, encode prompt, return session_id.

    Only valid if run is paused or completed.
    """
    from backend.training import artifacts, diagnostics
    from backend.training.templates import TEMPLATE_REGISTRY

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(
                db_run, "POST", "/api/training/{run_id}/diagnostics/start",
                {"prompt": req.prompt, "top_k": req.top_k, "max_prompt_tokens": req.max_prompt_tokens},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote diagnostics start failed: {exc}")
        await _touch_worker_for_run(db_run)
        remote_session_id = result.get("diagnostic_session_id")
        if remote_session_id:
            # Session itself lives in the trainer container's process — only
            # the id is recorded here so chatbot grounding knows which
            # session_id to ask the trainer about via the same _proxy path.
            diagnostics.record_session_for_run(run_id, remote_session_id)
        return result

    # Check run status
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
        raise HTTPException(400, "Run must be paused or completed to start diagnostics")

    # Check checkpoint exists
    cp_path = artifacts.checkpoint_path(run_id)
    if not cp_path.exists():
        raise HTTPException(400, "No checkpoint available for this run")

    rd = artifacts.run_dir(run_id)
    config = json.loads((rd / "config.json").read_text())
    template_key = config.get("template", "transformer")
    device = config.get("device", "cpu")

    # RNN's forward(x, hc) signature (one-hot input + threaded hidden state)
    # is fundamentally different from transformer/moe's forward(idx) — a
    # step-through diagnostic session for it needs its own hidden-state
    # bookkeeping, not yet built. Reject early and clearly rather than
    # crashing deeper in run_diagnostic_step(). RNN still gets a correct
    # static architecture manifest (see get_architecture_manifest) — only
    # the live step-through session is unsupported. See docs/DESIGN_DECISIONS.md.
    if template_key == "rnn":
        raise HTTPException(400, "Step-through diagnostics not yet supported for the RNN template — architecture view only.")

    try:
        # Load checkpoint
        import torch
        cp = torch.load(cp_path, map_location=device, weights_only=False)
        cp.pop("optimizer_state", None)
        model_config = cp.get("config", config)
        model = TEMPLATE_REGISTRY[template_key]["build_model"](model_config).to(device)
        model.load_state_dict(cp["model_state"])
        model.eval()

        # Load tokenizer
        if template_key in ("transformer", "moe"):
            from backend.training.templates.transformer.data import load_tiny_shakespeare, CharDataset
            text = load_tiny_shakespeare()
            tokenizer = CharDataset(text, config["model"]["block_size"], 1)
        elif template_key == "rnn":
            from backend.training.templates.rnn.data import load_dinos_dataset
            tokenizer = load_dinos_dataset(config["training"].get("seq_len", 50))
        else:
            raise HTTPException(400, f"Unknown template: {template_key}")

        # Encode prompt
        encoded = tokenizer.encode(req.prompt[:req.max_prompt_tokens])

        # Create session — temperature/decoding_mode read once here, same
        # source and values model.generate() (Generate button) uses, so
        # > / >> decode identically to Generate. See docs/DESIGN_DECISIONS.md.
        inference_cfg = config.get("inference", {})
        temperature = inference_cfg.get("temperature", 0.8)
        decoding_mode = inference_cfg.get("decoding_mode", "sample")
        session_id = diagnostics.create_diagnostic_session(
            model, tokenizer, device, encoded, run_id=run_id, temperature=temperature,
            decoding_mode=decoding_mode,
        )

        # Register hooks (delegates to model's register_diagnostic_hooks method)
        diagnostics.register_diagnostic_hooks(model, session_id)

        # Return session info with initial tokens
        input_tokens = [
            {"position": i, "id": tid, "text": tokenizer.decode([tid])}
            for i, tid in enumerate(encoded)
        ]

        return {
            "diagnostic_session_id": session_id,
            "tokens": input_tokens,
        }

    except Exception as e:
        training_log.error("Diagnostics start failed for run %d: %s", run_id, e, exc_info=True)
        raise HTTPException(500, f"Failed to start diagnostic session: {str(e)}")


@router.post("/{run_id}/diagnostics/{session_id}/step")
async def diagnostics_step(run_id: int, session_id: str, req: DiagnosticsStepRequest = None):
    """Execute one forward pass and return diagnostic snapshot.

    Only valid if run is paused or completed. Optional attention_layer/head
    (Phase 2) trigger an explicit attention capture for that layer/head.
    """
    from backend.training import artifacts, diagnostics

    attention_params = None
    if req is not None and req.attention_layer is not None and req.attention_head is not None:
        attention_params = (req.attention_layer, req.attention_head)
    qkv_detail = req.qkv_detail if req is not None else False
    attention_window_offset = req.attention_window_offset if req is not None else 0

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            body = {}
            if attention_params is not None:
                body = {
                    "attention_layer": attention_params[0], "attention_head": attention_params[1],
                    "qkv_detail": qkv_detail, "attention_window_offset": attention_window_offset,
                }
            result = await _proxy(db_run, "POST", f"/api/training/{{run_id}}/diagnostics/{session_id}/step", body)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote diagnostics step failed: {exc}")
        await _touch_worker_for_run(db_run)
        return result

    # Check run status
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
        raise HTTPException(400, "Run must be paused or completed for diagnostics")

    # Get session
    session = diagnostics.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Diagnostic session not found")

    # Run step
    snapshot = diagnostics.run_diagnostic_step(
        session_id, top_k=5, attention_params=attention_params, qkv_detail=qkv_detail,
        attention_window_offset=attention_window_offset,
    )
    if snapshot is None:
        raise HTTPException(500, "Failed to run diagnostic step")

    return snapshot.to_dict()


@router.post("/{run_id}/diagnostics/{session_id}/peek")
async def diagnostics_peek(run_id: int, session_id: str, req: DiagnosticsStepRequest = None):
    """Recompute the CURRENT state's snapshot with different attention
    params — no new token sampled, session/token_history untouched. Lets
    the UI refresh attention/Q-K-V immediately when the user changes Head
    in Inspector, instead of requiring a full step click. Real bug report,
    2026-07-14: "when I change head it should automatically show a
    different head" — mirrors diagnostics_step's local/remote dual path
    exactly, just calls run_diagnostic_step_internal (already existed, used
    internally by /generate's final-frame capture) instead of
    run_diagnostic_step. See docs/DESIGN_DECISIONS.md.
    """
    from backend.training import artifacts, diagnostics

    attention_params = None
    if req is not None and req.attention_layer is not None and req.attention_head is not None:
        attention_params = (req.attention_layer, req.attention_head)
    qkv_detail = req.qkv_detail if req is not None else False
    attention_window_offset = req.attention_window_offset if req is not None else 0

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            body = {}
            if attention_params is not None:
                body = {
                    "attention_layer": attention_params[0], "attention_head": attention_params[1],
                    "qkv_detail": qkv_detail, "attention_window_offset": attention_window_offset,
                }
            result = await _proxy(db_run, "POST", f"/api/training/{{run_id}}/diagnostics/{session_id}/peek", body)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote diagnostics peek failed: {exc}")
        await _touch_worker_for_run(db_run)
        return result

    status = artifacts.read_status(run_id)
    if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
        raise HTTPException(400, "Run must be paused or completed for diagnostics")

    session = diagnostics.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Diagnostic session not found")

    snapshot = diagnostics.run_diagnostic_step_internal(
        session_id, top_k=5, attention_params=attention_params, qkv_detail=qkv_detail,
        skip_token_generation=True, attention_window_offset=attention_window_offset,
    )
    if snapshot is None:
        raise HTTPException(500, "Failed to peek diagnostic state")

    return snapshot.to_dict()


@router.get("/{run_id}/diagnostics/{session_id}")
async def diagnostics_get(run_id: int, session_id: str):
    """Retrieve the last diagnostic snapshot (for reconnect/refresh).

    Does not advance the session.
    """
    from backend.training import artifacts, diagnostics

    db_run = await db.get_training_run(run_id)
    if _is_remote(db_run):
        try:
            result = await _proxy(db_run, "GET", f"/api/training/{{run_id}}/diagnostics/{session_id}")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Remote diagnostics get failed: {exc}")
        return result

    # Check run status
    status = artifacts.read_status(run_id)
    if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
        raise HTTPException(400, "Run must be paused or completed for diagnostics")

    # Get session and last snapshot
    session = diagnostics.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Diagnostic session not found")

    if session.last_snapshot is None:
        raise HTTPException(404, "No snapshots in this session yet")

    return session.last_snapshot.to_dict()


async def get_diagnostic_snapshot_for_run(run_id: int) -> dict | None:
    """Chatbot grounding accessor — latest diagnostic snapshot for a run, if
    a diagnostic session has been started for it. Reuses diagnostics_get()
    (the same local/remote _is_remote/_proxy dual path every other route
    here uses) rather than re-implementing it, so this stays correct for
    both local and serverless runs without duplicating that logic. Returns
    None (never raises) when there's no session yet, the run isn't
    paused/completed, or the run doesn't exist — the chatbot tool turns
    that into a plain "not available" message for the model.
    """
    from backend.training import diagnostics

    session_id = diagnostics.get_latest_session_id_for_run(run_id)
    if session_id is None:
        return None
    try:
        return await diagnostics_get(run_id, session_id)
    except HTTPException:
        return None


@router.post("/{run_id}/diagnostics/{session_id}/generate")
async def diagnostics_generate(run_id: int, session_id: str, req: DiagnosticsGenerateRequest):
    """Phase 3: continue generation (`>>`) — SSE stream, matching
    docs/fixtures/generate_stream.sample.txt. One `event: token` frame per
    generated token (no per-node capture — cost control), then one
    `event: done` frame with a full snapshot for the FINAL token only.
    """
    from backend.training import artifacts, diagnostics

    db_run = await db.get_training_run(run_id)

    async def event_stream():
        if _is_remote(db_run):
            try:
                endpoint_url = await _remote_endpoint_url(db_run)
                if endpoint_url is None:
                    raise HTTPException(502, "Remote worker endpoint not available")
                remote_path = f"/api/training/{db_run['remote_run_id']}/diagnostics/{session_id}/generate"
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream("POST", f"{endpoint_url}{remote_path}", json=req.model_dump()) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"
                await _touch_worker_for_run(db_run)
            except httpx.HTTPError as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        status = artifacts.read_status(run_id)
        if status is None or status.get("status") not in (RunStatus.PAUSED, RunStatus.COMPLETED):
            yield f"event: error\ndata: {json.dumps({'error': 'Run must be paused or completed'})}\n\n"
            return
        session = diagnostics.get_session(session_id)
        if session is None:
            yield f"event: error\ndata: {json.dumps({'error': 'Diagnostic session not found'})}\n\n"
            return

        try:
            import torch
            for _ in range(req.max_new_tokens):
                with torch.inference_mode():
                    all_tokens = session.prompt_tokens + session.token_history
                    idx = torch.tensor([all_tokens], dtype=torch.long, device=session.device)
                    if "Moe" in session.model.__class__.__name__:
                        logits, _, _ = session.model(idx)
                    else:
                        logits, _ = session.model(idx)
                    # Same recipe and same decoding_mode setting as
                    # model.generate() / _execute_forward_pass. See
                    # docs/DESIGN_DECISIONS.md.
                    if session.decoding_mode == "greedy":
                        next_id = torch.argmax(logits[0, -1, :]).item()
                    else:
                        sample_probs = torch.softmax(logits[0, -1, :] / session.temperature, dim=-1)
                        next_id = torch.multinomial(sample_probs, num_samples=1).item()
                    session.token_history.append(next_id)
                    session.generation_step += 1

                yield f"event: token\ndata: {json.dumps({'position': len(session.prompt_tokens) + len(session.token_history) - 1, 'id': next_id, 'text': session.tokenizer.decode([next_id]), 'generation_step': session.generation_step})}\n\n"

            attention_params = None
            if req.attention_layer is not None and req.attention_head is not None:
                attention_params = (req.attention_layer, req.attention_head)
            final_snapshot = diagnostics.run_diagnostic_step_internal(
                session_id, top_k=5, attention_params=attention_params,
                qkv_detail=req.qkv_detail, skip_token_generation=True,
            )
            if final_snapshot is None:
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to capture final snapshot'})}\n\n"
                return

            # Phase 4: persist the final outcome once the stream completes —
            # not per-token, per contract cost-control guidance.
            prompt_text = session.tokenizer.decode(session.prompt_tokens)
            generated_output = session.tokenizer.decode(session.prompt_tokens + session.token_history)
            generation_params = {
                "max_new_tokens": req.max_new_tokens,
                "attention_layer": req.attention_layer,
                "attention_head": req.attention_head,
                "qkv_detail": req.qkv_detail,
            }
            await db.save_diagnostic_session_result(
                run_id,
                prompt=prompt_text,
                generated_output=generated_output,
                generation_params=generation_params,
                top_k_summary=final_snapshot.lm_head.get("top_k", []),
            )

            yield f"event: done\ndata: {json.dumps({'final_snapshot': final_snapshot.to_dict()})}\n\n"
        except Exception as e:
            training_log.error("Diagnostics generate failed for run %d: %s", run_id, e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
