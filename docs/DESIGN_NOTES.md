# Design Notes: Known Issues & Architecture Decisions

## GPU Training + Web UI Responsiveness (GIL Problem)

### The Problem

When training on GPU through the web UI, the dashboard may become unresponsive:
- Step counter stops updating
- Loss curves don't appear
- Pause/Stop buttons don't respond
- In severe cases, the entire terminal/VS Code freezes

This does **not** happen when running training standalone (e.g. `gpu_diag.py`).

### Root Cause: Python's Global Interpreter Lock (GIL)

The web app runs training in a background `threading.Thread`, while FastAPI handles HTTP requests in the main thread. Python's GIL only allows one thread to execute Python code at a time.

During training:
- PyTorch releases the GIL during CUDA kernel execution (the actual GPU math)
- But between kernels — batch loading, tensor indexing, Python loops, `loss.item()` — the training thread **holds the GIL**
- During eval phases (tight loop of forward passes), GIL is held almost continuously
- FastAPI's event loop can't process status/metrics requests
- UI appears frozen

This is a fundamental limitation of Python threading for CPU-bound work. It's the same reason Ctrl+C can feel slow to interrupt training in Colab or Jupyter.

### Current Mitigation

We add `time.sleep(0)` after each training step and inside eval loops. This explicitly yields the GIL, giving FastAPI a chance to process requests. It adds negligible overhead (~0.1%) but keeps the UI responsive enough for status updates and pause/stop to work.

### Proper Fix (Future Work)

Replace `threading.Thread` with `multiprocessing.Process` for training runs. A separate process has its own GIL, so training and the web server never contend. This requires:

1. Inter-process communication for metrics (shared memory, pipes, or writing to DB)
2. Process management for pause/resume (signals instead of threading.Event)
3. Careful GPU memory cleanup when stopping a process

Estimated effort: ~2 days of refactoring `runner.py` and `training.py` API.

Alternative: Run training as a subprocess (`subprocess.Popen`) executing a standalone script. Read metrics from the JSONL file on disk. Simpler but less integrated.

### What Collaborators Need to Know

1. **CPU mode works reliably** — GIL contention is minimal because training is slower and yields naturally
2. **GPU mode works but UI may lag** — training completes correctly, metrics are saved, but dashboard updates may be delayed
3. **Don't panic if UI freezes during GPU training** — the training IS running. Wait for it to finish or use `pkill -f uvicorn` from another terminal
4. **Standalone script for serious GPU runs** — use `gpu_diag.py` as a template for GPU training without the web UI
5. **Never use `--reload` with GPU training** — hot-reload kills the training process and may leave GPU memory in a bad state

### Diagnostic Results (2026-06-26)

Standalone GPU training on RTX PRO 2000 Blackwell (8GB VRAM):
- 2500 steps completed in 4 minutes (95 ms/step)
- VRAM: 92 MB steady, zero memory leak
- Loss: train 4.2 -> 1.32, val 4.2 -> 1.56
- See `gpu_diag_results.txt` for full log

## Training Speed Reference

| Device | Steps/min | Time for 5000 steps |
|--------|-----------|-------------------|
| CPU (i7/Ryzen) | ~25-50 | 100-200 min |
| RTX PRO 2000 (8GB) | ~630 | ~8 min |
| Cloud GPU (A100) | ~3000+ | <2 min |

## Eval Configuration

Default eval settings are tuned for CPU responsiveness:
- `eval_interval`: 100 steps (how often to compute validation loss)
- `eval_iters`: 10 (number of batches per eval — capped at 10 in runner.py)

Lower `eval_iters` = noisier loss estimates but faster eval. For this tiny model, 10 iters gives sufficient signal.

## Known Test Limitations

### RNN Pause Timing (3 test failures)

The integration test (`tests/test_integration_2026_06_24.py`) has 3 pre-existing failures related to RNN pause/resume timing:

- `rnn paused` — expects `paused` status but gets `running`
- `rnn resume` — fires while still running (not yet paused)
- `rnn resumed running` — expects `running` but gets `paused`

**Root cause:** Cooperative pause uses `threading.Event`. The test sleeps 3 seconds after sending pause, but on slower machines or under load, the training thread may not reach the next `_check_pause()` call within that window.

**These failures do NOT indicate a bug.** Pause/resume works correctly — the test timing is just tight. If you increase the sleep from 3s to 8-10s, all three pass. We keep the shorter sleep to avoid slow test runs.

**Workaround for reviewers:** Run the test twice, or increase the pause sleep. The 34 other tests cover all API endpoints, exports, training start/stop, and error cases.

---

## Tier 2: Before Production

The following are documented design gaps acceptable for Tier 1 (local prototype) but should be addressed before production/cloud deployment.

### Config Validation
- Backend accepts `config: dict` (unvalidated). Should add Pydantic models per template (TransformerConfig, RNNConfig) to validate keys, types, and numeric bounds.
- Frontend uses `Record<string, number | string>` — should match backend schemas with specific TypeScript types.
- The eval_iters cap (silently capped to 10 in runner.py) should be surfaced in the UI or enforced via config validation.

### Run Metadata & Reproducibility
Durable run records (`training_runs` table) currently lack:
- Config snapshot (exact config used, not just experiment's current config)
- Seed used
- Template key and dataset name
- Checkpoint and metrics file paths
- Error message on failure
- Device name / CUDA device info
- Package versions and git commit
- Run mode (thread/subprocess/docker/cloud)
- Cancellation/timeout reason

### Richer Status States
Current: `queued → running → paused → completed/failed`

Production needs: `queued → starting → running → pause_requested → checkpointing → paused → resuming → completed → failed → cancelled → timeout`

### Idempotency
- `POST /training/start` — no duplicate-run protection. Should check for already-running run per (experiment_id, device), or accept an idempotency key.
- `POST /experiments/from-preset/{key}` — always creates new row. Double-click creates duplicates.
- Metric writes — no uniqueness on (run_id, step). Retries duplicate rows.
- Pause/resume/stop — return 400 if already in target state instead of treating as idempotent no-ops.
- `max_concurrent_runs` exists in settings but is not enforced.

### SQL Safety
`update_training_run()` and `sync_update_training_run()` build column names from kwargs. Values are parameterized (safe), but column names are not whitelisted. Low risk since kwargs are internal-only, but should add an allowed-column set before exposing any user-controlled field names.

### Frontend Error Handling
- Polling swallows errors silently (`catch { }`)
- Config autosave does not surface failures to the user
- Notes autosave has no try/catch
- If backend goes down, UI silently drifts out of sync

### Export System
`export.py` duplicates model/training logic as large f-string templates rather than reusing the template modules. This means training logic exists in three places: runner.py, templates/, and export.py. Should refactor toward composing exports from the actual template source files.

### Logging Gaps
Missing from logs:
- Request/correlation IDs
- Config diff on update
- Device availability/CUDA details
- Frontend errors
- Status log previously showed "?" for old status (fixed)

### Metrics Enrichment
Currently captures train_loss and val_loss per eval point. Should eventually add:
- Learning rate, tokens/sec, examples/sec
- GPU memory allocated/reserved
- Validation perplexity
- Elapsed wall-clock time per metric point
- Parameter count
- Sample generated outputs at checkpoints

### Runner Decomposition
`runner.py` (390 lines) handles state, checkpoints, metrics, status, two training loops, pause/resume/stop, and prompt inference. When subprocess/cloud orchestration is added, split into: `run_state.py`, `artifacts.py`, `local_thread_runner.py`, `train_transformer.py`, `train_rnn.py`, `prompting.py`.

### Preset DRY-up
Three transformer presets repeat most fields, varying only name, pos_encoding, and learning_rate. Define a base transformer config and create variants with shallow overrides as presets grow.

## File Layout

See README.md for project structure and setup instructions.
