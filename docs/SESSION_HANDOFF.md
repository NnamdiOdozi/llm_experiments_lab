# PROJECT STATE — Session 2026-06-28 (Phase 3 complete)

## 1. Objective
LLM Experiments Lab: browser-based training tool. FastAPI + React/TS. Three templates: transformer, MoE, RNN.

## 2. What was done this session
### Phase 1 (frontend resilience) — completed
- React ErrorBoundary component wrapping both App views
- Richer polling: lastPollSuccess timestamp, pollError, stale data warning (>10s)
- 429 handling in handleStart() — "Max concurrent runs reached" banner
- startError state displayed in TrainingControls

### Phase 2 (DB/disk-backed status) — completed
- `db.get_run_status_from_db()` — reads training_runs table, reconstructs status dict
- `GET /status` falls back to DB when run not in memory
- `GET /metrics` reads metrics.jsonl from disk when run not in memory
- Enables visibility of past runs after backend restart

### Phase 3 (subprocess training worker) — completed
- **NEW: `backend/training/artifacts.py`** — run directory layout, flag files, status.json read/write
- **NEW: `backend/training/train_worker.py`** — standalone training script (transformer/MoE/RNN), file-based communication
- **REWRITE: `backend/training/runner.py`** — orchestrator using subprocess.Popen, no more threading/in-memory models
- **UPDATED: `backend/api/training.py`** — _count_active_runs checks process liveness, websocket reads from disk
- **UPDATED: tests** — prompt test expects 400 (subprocess mode), poll-based transformer startup
- Training now runs in separate process. FastAPI stays responsive. No GIL contention.
- Pause/resume works via flag files. Stop via flag file.
- Prompt-while-paused returns 400 (model in worker process) — Phase 5 will address.

## 3. What to do next
See `docs/SUBPROCESS_MIGRATION_PLAN.md`:
- Phase 4: Checkpoint-based pause/resume (pause flag → worker saves checkpoint + exits, resume = new worker from checkpoint)
- Phase 5: Prompt-from-checkpoint (load checkpoint in API or spawn inference subprocess)
- Phase 6: Docker containerization

## 4. Key files modified/created
- `backend/training/artifacts.py` (NEW) — paths, flags, status.json
- `backend/training/train_worker.py` (NEW) — standalone training loops
- `backend/training/runner.py` (REWRITE) — subprocess orchestrator
- `backend/api/training.py` — process-based concurrency, disk-based metrics/websocket
- `backend/db.py` — get_run_status_from_db()
- `frontend/src/components/ErrorBoundary.tsx` (NEW)
- `frontend/src/App.tsx` — ErrorBoundary, polling state, 429 handling
- `frontend/src/components/TrainingControls.tsx` — stale warning, startError display
- `tests/test_integration_2026_06_24.py` — adapted for subprocess mode

## 5. Commands
```bash
cd /home/nodozi/projects/NEBIUS_MAR_2026/Nebius_serverless/llm_experiments_lab
nohup .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > /tmp/uvicorn.log 2>&1 &
cd frontend && npm run dev  # Vite on 5173
.venv/bin/python tests/test_integration_2026_06_24.py  # 38 tests
```

## 6. Architecture (current)
```
FastAPI process (API only, no training)
  ├── API handlers
  ├── Reads: DB + data/runs/{id}/status.json + metrics.jsonl
  └── Writes: stop.flag, pause.flag, config.json

Training subprocess (per run)
  ├── Owns: model, optimizer, dataset (in-process)
  ├── Writes: metrics.jsonl, checkpoint.pt, status.json, run_meta.json
  └── Watches: stop.flag, pause.flag
```

## 7. Known issues
- Prompt-while-paused not available (returns 400) — Phase 5
- Parent repo .gitignore has `temp*` matching `templates/` — needs fix to `temp_*`
- Parent repo .gitignore has `tests/` — ignores all test dirs
- No PUT endpoint for experiments
- export.py doesn't include MoE model code for standalone export

## 8. What NOT to do
- Don't delete .db files without asking
- Don't run GPU + CPU training concurrently on this laptop (8GB VRAM, 15GB RAM)
- Don't use `git add -A` without checking .gitignore first
- Don't background server processes with `&` from Claude terminal
- Don't try to access model from API process — model lives in worker subprocess
