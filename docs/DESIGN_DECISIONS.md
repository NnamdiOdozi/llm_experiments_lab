# Design Decisions

Non-obvious decisions, tricky bug fixes, and architecture context — documented to prevent regressions.

---

## 1. Architecture: Subprocess-Based Training

### Problem (Historical)

The original architecture ran training in a `threading.Thread` alongside FastAPI in the main thread. Python's GIL caused the dashboard to freeze during GPU training — the training thread held the GIL between CUDA kernels (batch loading, tensor indexing, `loss.item()`), starving FastAPI's event loop.

### Solution (Implemented)

Training runs in a separate **subprocess** (`subprocess.Popen`) executing `backend.training.train_worker`. Communication is via files:
- `status.json` — current step, status, elapsed time
- `metrics.jsonl` — per-step training/eval metrics
- Flag files (`stop`, `pause`) — cooperative signal mechanism

This gives training its own GIL, so the API server never contends with training.

### What Collaborators Need to Know

1. **CPU mode works reliably** — even the old threading approach had minimal contention on CPU
2. **GPU mode works** — subprocess isolation eliminates GIL contention entirely
3. **Never use `--reload` with GPU training** — hot-reload kills the training process and may leave GPU memory in a bad state
4. **Standalone script for debugging** — use `tools/gpu_probe.py` for GPU training diagnostics without the full web UI

---

## 2. Cross-Platform Compatibility (Windows / WSL / Linux)

### Problem

The codebase was developed on WSL2 and used Linux-only APIs:
- `preexec_fn` in `subprocess.Popen` — raises `ValueError` on Windows
- `prctl(PR_SET_PDEATHSIG)` — Linux-only syscall
- Hardcoded `/usr/lib/wsl/lib/nvidia-smi` — WSL2-specific path
- `pgrep`, `free`, `uptime`, `ps` — Linux-only commands in diagnostic tools
- `.venv/bin/` — Linux venv layout (Windows uses `.venv\Scripts\`)

### Solution

Runtime platform detection via `sys.platform == "win32"` at each divergence point:

| Component | Linux/WSL | Windows |
|-----------|-----------|---------|
| Worker subprocess | `preexec_fn=_set_pdeathsig` | `creationflags=CREATE_NEW_PROCESS_GROUP` |
| nvidia-smi | `shutil.which()` → WSL fallback path | `shutil.which()` (finds it in PATH) |
| Venv paths | `.venv/bin/python3` | `.venv\Scripts\python.exe` |
| Process kill | `pgrep` + `kill` | `wmic` + `taskkill` |
| System diagnostics | `free`, `uptime`, `ps` | `wmic`, `powershell` equivalents |

**Key files:** `runner.py` (`_popen_kwargs()`), `tools/compat.py` (shared helpers).

**DO NOT:** Add platform-specific code paths without the `sys.platform` guard. All code must run on both platforms from a single codebase.

---

## 3. GPU Yield: Time-Based, Not Step-Based

### Problem

Dashboard freezes during fast GPU training on WSL2. Steps and charts show zero updates, then jump 400–500 steps when Pause is pressed.

### Root Cause

WSL2's display compositor shares the GPU with training via GPU-PV (paravirtualization). The old step-based yield (`cuda.synchronize() + sleep(1ms)` every 10 steps) was too short for the compositor to render a frame (~16.7ms at 60fps). Once the GPU hits full speed, the browser literally cannot repaint.

**Why it looks like a freeze:** Frontend polling IS working, React IS updating state, but the browser window cannot repaint. When Pause stops the GPU, the compositor catches up and renders all buffered state changes at once — big step jump.

### Solution

Time-based yield in `WorkerState.yield_gpu()`:
- Fires every `gpu_yield_interval_sec` (100ms wall time), not every N steps
- Sleeps `gpu_yield_sleep` (20ms) — enough for one compositor frame
- Also calls `update_progress()` to write fresh `current_step` to status.json
- Adapts to training speed: barely fires on slow CPU, fires regularly on fast GPU
- Overhead: ~17% (20ms/120ms) — acceptable for a lab environment

### Settings (`config/settings.py`)
- `gpu_yield_interval_sec: float = 0.1`
- `gpu_yield_sleep: float = 0.02`

**DO NOT:** Revert to step-based yield or reduce the sleep below 16ms. The compositor needs a full frame to render.

### Note on Native Windows

On native Windows (not WSL2), GPU-PV is not involved — the GPU driver serves CUDA and display independently. The yield mechanism still works but is less critical. It remains enabled for consistent behavior and to keep `status.json` updates regular.

---

## 4. Performance Reference

### Training Speed

| Device | Steps/min | Time for 5000 steps |
|--------|-----------|---------------------|
| CPU (i7/Ryzen) | ~25–50 | 100–200 min |
| RTX PRO 2000 (8GB) | ~630 | ~8 min |
| Cloud GPU (A100) | ~3000+ | <2 min |

### Eval Configuration

Default eval settings are tuned for responsiveness:
- `eval_interval`: 20 steps (how often to compute validation loss)
- `eval_iters`: 2 (number of batches per eval)

Lower `eval_iters` = noisier loss estimates but faster eval. For this tiny model, 2 iters gives sufficient signal.

---

## 5. Known Test Limitations

### RNN Pause Timing (3 test failures)

The integration test (`tests/test_integration_2026_06_24.py`) has 3 pre-existing failures related to RNN pause/resume timing:

- `rnn paused` — expects `paused` status but gets `running`
- `rnn resume` — fires while still running (not yet paused)
- `rnn resumed running` — expects `running` but gets `paused`

**Root cause:** Cooperative pause uses flag files. The test sleeps 3 seconds after sending pause, but on slower machines or under load, the training worker may not reach the next flag check within that window.

**These failures do NOT indicate a bug.** Pause/resume works correctly — the test timing is just tight. Increasing the sleep from 3s to 8–10s makes all three pass.

---

## 6. Tier 2: Before Production

Design gaps acceptable for local prototype but should be addressed before production/cloud deployment.

### Config Validation
- Backend accepts `config: dict` (unvalidated). Should add Pydantic models per template (TransformerConfig, RNNConfig) to validate keys, types, and numeric bounds.
- Frontend uses `Record<string, number | string>` — should match backend schemas with specific TypeScript types.

### Idempotency
- `POST /training/start` — no duplicate-run protection. Should check for already-running run per (experiment_id, device).
- `POST /experiments/from-preset/{key}` — always creates new row. Double-click creates duplicates.
- Metric writes — no uniqueness on (run_id, step). Retries duplicate rows.
- Pause/resume/stop — return 400 if already in target state instead of treating as idempotent no-ops.
- `max_concurrent_runs` exists in settings but is not enforced.

### SQL Safety
`update_training_run()` and `sync_update_training_run()` build column names from kwargs. Values are parameterized (safe), but column names are not whitelisted. Low risk since kwargs are internal-only, but should add an allowed-column set before exposing any user-controlled field names.

### Frontend Error Handling
- Polling swallows errors silently (`catch { }`)
- Config autosave does not surface failures
- Notes autosave has no try/catch
- If backend goes down, UI silently drifts out of sync

### Export System
`export.py` duplicates model/training logic as large f-string templates rather than reusing the template modules. Training logic exists in three places: runner.py, templates/, and export.py. Should refactor toward composing exports from actual template source files.

### Preset DRY-up
Three transformer presets repeat most fields, varying only name, pos_encoding, and learning_rate. Define a base transformer config and create variants with shallow overrides as presets grow.

---

## 7. Chatbot Audit-Log Matching is a Substring Check, Not Structured Data

`backend/chatbot/context.py::_get_last_audit_change()` finds the most
recent config/experiment change for the grounded chatbot by searching the
current session's log file for the substring `"id=<experiment_id> "`.
This works because every `audit_log.info(...)` call in
`backend/api/experiments.py` formats its message as either `id=%d ...` or
`experiment_id=%d ...`, and both forms end in `id=<N> ` followed by more
text — so one substring check covers both.

**This is coupled to the exact audit log message format.** If a future
`audit_log.info(...)` call site is added or changed such that the
experiment ID is no longer immediately followed by a space (e.g. it's
last on the line, or formatted as `id: 5` instead of `id=5`), the
chatbot will silently stop finding "last change" for that log line — no
error, just an empty/stale result. If you touch audit log message
formats, check `_get_last_audit_change()`'s tests
(`tests/test_chatbot_context.py`) still pass.

The same coupling applies to pause-and-prompt visibility:
`backend/api/training.py::prompt_model` logs each exchange as
`run_id=<N> payload=<json>` (full prompt + full output + step, JSON-encoded
so embedded newlines can't break the one-line-per-record format), and
`context.py::_get_prompt_history()` parses those lines back out for the
Lab Assistant. If you change that log line's format, update both sides.
Like the audit lookup, this only sees the current server session's log —
prompt exchanges from before a backend restart are invisible to the chatbot
(accepted v1 tradeoff, same as "last change" tracking).

## 8. `nebius` CLI Calls Must Close stdin and Be Timeout-Bounded

**Incident (2026-07-11):** a live smoke test of the endpoint track hung
indefinitely. `POST /api/training/start` never returned. Root cause:
`backend/nebius/endpoints_client.py::_run_cli()` spawned the `nebius` CLI
via `asyncio.create_subprocess_exec` without setting `stdin=`, so the
child inherited uvicorn's stdin — which, running as a background process,
never produces EOF. The `nebius ai endpoint start` subprocess sat blocked
waiting for input that would never arrive. Running the exact same CLI
command directly in a terminal with `< /dev/null` returned in seconds
(with an unrelated transient API error), confirming the hang was
stdin-related, not a slow API call.

**Fix:** every `_run_cli()` call now passes `stdin=asyncio.subprocess.DEVNULL`
explicitly, and the whole call is wrapped in `asyncio.wait_for(...,
timeout=settings.nebius_cli_timeout_seconds)` as a second line of defense
— if some other future CLI call hangs for a different reason, it fails
loudly after `nebius_cli_timeout_seconds` (default 60s) instead of hanging
the request forever. On timeout the subprocess is explicitly killed
(`proc.kill()` + `await proc.wait()`) so it doesn't leak as a zombie.

**If you add another subprocess call anywhere in this codebase** (CLI
wrapper, external tool, etc.), close stdin explicitly rather than relying
on the default inherited-from-parent behavior — it works fine when run
interactively or under pytest, and only breaks under a real long-running
server process, which makes it easy to miss until it happens in production.

Separately: `nebius_endpoint_ready_timeout_seconds` (the poll-loop timeout
for "wait until the endpoint reaches RUNNING") was bumped from 180s to
360s after being told endpoint creation can take up to ~5 minutes in
practice — noticeably longer than the plan doc's 30-90s estimate.

---

## File Layout

See `README.md` for project structure and setup instructions.
