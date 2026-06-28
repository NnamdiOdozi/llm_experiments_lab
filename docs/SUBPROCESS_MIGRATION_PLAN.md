# Subprocess Migration Plan

## Status Key
- [ ] Not started
- [~] In progress
- [x] Done

---

## Phase 0: Correctness fixes (pre-architecture)
- [x] Fix checkpoint-before-release ordering (all 3 loops)
- [x] Enforce max_concurrent_runs (max 2 total, max 1 GPU)
- [x] Periodic checkpoint saving (every eval_interval)
- [x] Memory cleanup on run end (_release_heavy_objects)
- [x] Pre-eval pause check (faster cooperative pause)
- [x] MoE code view + prompt support
- [x] .gitignore: *.pt, *.db, data/

## Phase 1: Frontend resilience (~130 lines)
- [x] Backend disconnected banner (network errors, not 404)
- [x] sessionStorage per-tab (not localStorage)
- [x] React ErrorBoundary component (~30 lines)
  - New: `frontend/src/components/ErrorBoundary.tsx`
  - Modify: `App.tsx` — wrap root
- [x] Richer polling/error display (~50 lines)
  - Track: lastPollSuccess timestamp, pollError string
  - Show: last successful poll time, stale data warning, run ID
  - Modify: `App.tsx`, `TrainingControls.tsx`
- [x] Handle 429 (concurrency limit) in frontend start handler (~15 lines)
  - Show: "Max runs reached. Stop a run first."
  - Modify: `App.tsx` handleStart()

### Phase 1 files:
- `frontend/src/components/ErrorBoundary.tsx` (NEW)
- `frontend/src/App.tsx`
- `frontend/src/components/TrainingControls.tsx`

---

## Phase 2: File/DB-backed status & metrics (~60 lines)

Goal: API reads from durable storage, not active_runs dict.
Enables: backend restart without losing run visibility.

- [x] GET /status reads from DB first, falls back to active_runs
  - `backend/api/training.py` (~20 lines)
- [x] GET /metrics reads from metrics.jsonl on disk
  - `backend/api/training.py` (~20 lines)
- [x] DB helper: get_run_status_from_db()
  - `backend/db.py` (~20 lines)

### Phase 2 files:
- `backend/api/training.py`
- `backend/db.py`

---

## Phase 3: Subprocess training worker (~500 lines)

Goal: Training runs in separate process. FastAPI stays responsive.

### Included:
- [x] `backend/training/train_worker.py` (~150 lines)
  - Standalone script: receives config via JSON file/arg
  - Writes: metrics.jsonl, checkpoint.pt, status.json, run_meta.json
  - Watches: stop flag file
  - Exits: on completion, error, or stop signal
- [x] `backend/training/artifacts.py` (~60 lines)
  - Run directory layout: paths, status file read/write, flag files
  - Shared between worker and API
- [x] Modify `runner.py` to orchestrator role (~200 lines rewrite)
  - start_run() → subprocess.Popen(train_worker.py)
  - active_runs tracks PID + reads status from files
  - stop_run() → write stop flag or send signal
  - Remove: threading, in-memory model/dataset/optimizer
- [x] Modify `backend/api/training.py` (~30 lines)
  - Status/metrics from files (Phase 2 already done)
  - Stop via flag file

### NOT included:
- Pause/resume (Phase 4)
- Prompt while paused (Phase 5)
- Docker (Phase 6)

### Phase 3 files:
- `backend/training/train_worker.py` (NEW)
- `backend/training/artifacts.py` (NEW)
- `backend/training/runner.py` (REWRITE)
- `backend/api/training.py`

---

## Phase 4: Checkpoint-based pause/resume (~80 lines)

Requires: Phase 3 (subprocess)

- [x] Write pause flag file → worker detects, saves checkpoint, exits
- [x] API marks status "paused" + records checkpoint path
- [x] Resume = start new worker from checkpoint (--resume flag)
- [x] Frontend pause/resume unchanged (already status-driven)

### Phase 4 files:
- `backend/training/train_worker.py`
- `backend/training/runner.py`
- `backend/api/training.py`

---

## Phase 5: Prompt-from-checkpoint (~100 lines)

Requires: Phase 4 (checkpoint-based pause)

Options (pick one):
- A) Load checkpoint in API process temporarily (simple, blocks API)
- B) Spawn inference subprocess (clean, more complex)
- C) Keep training subprocess alive in paused state (hybrid)

- [x] Implement option A (load checkpoint in API process)
- [x] PausePrompt.tsx unchanged (already sends POST /prompt)

### Phase 5 files:
- `backend/api/training.py`
- `backend/training/runner.py`
- Maybe: `backend/training/infer_worker.py` (NEW, if option B)

---

## Phase 6: Docker (later)

Requires: Phase 3 stable

- [ ] Dockerfile for train_worker.py
- [ ] Docker-based start_run() in runner.py
- [ ] Volume mounts for data/runs/
- [ ] Container lifecycle management

---

## Architecture before/after

### Current (threaded, in-process):
```
FastAPI process
  ├── API handlers (main thread)
  ├── Training thread 1 (model + optimizer in memory)
  ├── Training thread 2 (model + optimizer in memory)
  └── active_runs dict (metrics, status, model refs)
```

### Target (subprocess, file-backed):
```
FastAPI process
  ├── API handlers
  ├── Reads: DB + data/runs/{id}/status.json + metrics.jsonl
  └── Writes: stop/pause flag files

Training subprocess 1
  ├── Owns: model, optimizer, dataset
  ├── Writes: metrics.jsonl, checkpoint.pt, status.json
  └── Watches: stop/pause flag files

Training subprocess 2
  └── (same)
```
