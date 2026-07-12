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

## 9. Worker Spec Display Must Read the Real Endpoint, Never the Config

**Incident (2026-07-11):** `config/settings.py::nebius_cpu_preset` was bumped
from `4vcpu-16gb` to `8vcpu-32gb`, and the frontend's worker tag started
showing `8vCPU / 32GB` — but the actual running endpoint was still the
original `4vcpu-16gb` one, never recreated. The tag was silently wrong,
actively misleading about what compute is really running.

Root cause: `backend/api/nebius.py::get_worker_status()` returned
`settings.nebius_cpu_preset`/`nebius_gpu_preset` directly — the config value
that would be used for a *future* `create_endpoint` call, not what any
*existing* endpoint is actually running. A settings change doesn't
retroactively resize a live endpoint; compute specs are fixed at creation
time.

**Fix:** `worker_sessions` gained `actual_platform`/`actual_preset` columns.
`worker_manager.py::ensure_worker()` captures these from the endpoint's own
`spec.platform`/`spec.preset` (the CLI's real answer, from `nebius ai
endpoint get`) at the moment it confirms RUNNING, and `get_worker_status()`
now returns that stored value — `None` if no endpoint has ever actually
been provisioned, never a config-based guess. The frontend tag renders
`CPU · Serverless` (no spec segment) until a preset value is genuinely
known, then `CPU · 4vCPU / 16GB · Serverless` once it is.

**General lesson:** any UI field claiming to describe live infrastructure
state must be sourced from that infrastructure's own reported state, not
from the config that was used (or intended) to create it. Config describes
intent; only the provider's API describes reality.

## 10. Per-Run Backend Must Come From the Run, Not Global Config

**Incident (2026-07-11):** a screenshot showed a run tagged `CPU · Serverless`
with the idle-timeout banner simultaneously claiming the worker was stopped
due to inactivity — visibly contradictory. Checked the DB directly: the run
in question had `execution_backend='local'`, but the tag showed "Serverless"
anyway.

Root cause: `TrainingControls`'s worker tag and `WorkerIdleBanner` both read
`GET /api/nebius/workers/{device}` — the app's *current global*
`training_backend` setting — completely independent of which
`execution_backend` the specific run being viewed actually used. A run
started under `local` mode keeps `execution_backend='local'` on its DB row
forever, but if the server was later restarted with
`TRAINING_BACKEND=nebius_endpoint`, every run's tag would start showing
"Serverless" regardless of what that run actually used — exactly the
class of bug as §9, one level up: config was standing in for a fact that
can only be true per-instance (there, per-endpoint; here, per-run).

**Fix:** every run-status response (`backend/training/runner.py::get_run_status()`,
`backend/db.py::get_run_status_from_db()`, and the remote-proxy override in
`backend/api/training.py::run_status()`) now carries `execution_backend`
reflecting *that run's own* value — for local runs this is trivially
`"local"` (a subprocess's own status.json has no concept of remote
execution), for remote runs the controller explicitly overrides whatever
the proxied endpoint's response claims (its own status.json also always
says "local" from its own point of view, which is correct for it but wrong
from the controller's). The frontend now branches on
`runStatus.execution_backend`, not a global settings poll, and
`WorkerIdleBanner` is hidden entirely when the current run is definitively
local — showing remote-worker idle status while looking at a local run is
noise, not signal.

**General lesson, extending §9:** "current global config" and "the actual
backend a specific already-started thing is using" are different facts that
can diverge the moment either one changes after the fact. Anywhere both
exist, the per-instance value must win for display — global config is only
a default for *new* things, never a description of *existing* ones.

## 11. `training_backend` Is a Per-Request Choice, Not a Global Switch

**Change (2026-07-11), following directly from §10:** `training_backend`
in `config/settings.py` used to be the *only* way to pick local vs.
Nebius-endpoint execution — set once via `TRAINING_BACKEND` env var at
server startup, applying to every run for that process's lifetime. Restart
the server with a different value to change it; there was no way to pick
per run.

This didn't match how `device` (CPU/GPU) already worked — that's chosen on
the frontend per experiment, no restart needed — and users reasonably
expected the same for local-vs-serverless, especially after §10 made clear
that different runs legitimately use different backends over a session's
lifetime.

**Fix:** `StartRunRequest` (`backend/api/training.py`) gained a `backend`
field, defaulting to `"local"`. `POST /api/training/start` branches on
`req.backend`, not `settings.training_backend` — the setting is now only
the frontend dropdown's pre-selected value on first load, never consulted
by the backend once a request carries its own explicit choice. Frontend:
`PresetPicker` (landing page) and `TrainingControls` (in-workspace, next to
the existing device selector, shown between runs) both gained a matching
Backend dropdown, mirroring exactly how device selection already works in
both places.

**Note:** `settings.training_backend` itself still exists but is now
effectively dead as a runtime switch — nothing reads it to decide a run's
backend anymore. Kept for now as a documented historical field / possible
future default-seeding use; safe to remove if that never materializes.

## 12. `Dockerfile.trainer-cpu`'s torch Swap — Four Chained Bugs Found by Actually Building It

**Context:** `Dockerfile.trainer-cpu` swaps torch from the CUDA wheel
(installed by `uv sync` against `pyproject.toml`'s pinned cu130 index) to
the CPU-only wheel. All four issues below were found by actually running
`docker build` / `docker run` against the real Dockerfile (2026-07-12),
not by reasoning about it — each one was silent (no error, or a confusing
unrelated-looking error) until specifically checked for.

**a) `--reinstall-package` needs `--no-deps`.** `uv pip install
--index-url <cpu-index> --reinstall-package torch torch` without
`--no-deps` silently re-resolved and changed *other*, unrelated packages'
versions too (in an isolated test: `urllib3`/`charset-normalizer` shifted
when only `requests` was named for reinstall). Fix: always pair
`--reinstall-package <name>` with `--no-deps` when the intent is "swap
only this one package." Verified with `--no-deps` that every other
package's version is provably untouched.

**b) The read-back version string carries a local-build suffix.** The
Dockerfile reads back the exact torch version `uv sync` installed (`uv pip
show torch`) so the CPU and GPU images can never drift onto different
torch versions of each other. But uv reports it as `2.12.1+cu130` — that
`+cu130` is a PEP 440 local version identifier specific to the cu130
index. Pinning `torch==2.12.1+cu130` against the *CPU* index is
unsatisfiable (confirmed via a real failed `docker build`, not a guess).
Fix: `cut -d'+' -f1` to strip the suffix before reuse.

**c) `uv run` (used as `CMD`) silently reverts the swap on every
container start.** `uv run` always reconciles the venv against `uv.lock`
before running anything. `uv.lock` still resolves torch to the cu130
build, so on every cold start it detected "drift" from the CPU swap and
reinstalled cu130 torch back over it — confirmed by running the container
and checking `torch.__version__`, which came back `2.12.1+cu130` after a
plain `uv run python -c "import torch"`. This would have cost ~1 minute
per cold start too, which matters a lot for an idle-timeout endpoint that
stops and restarts often. Fix: `CMD ["uv", "run", "--no-sync", ...]` —
runs against the venv exactly as baked, skips the reconcile check. Applied
to both Dockerfiles for consistency, though only load-bearing on the CPU
one.

**d) Orphaned CUDA packages + uv's own wheel cache bloat the image even
after the swap.** `uv sync` installs the *full* cu130 dependency tree
first (nvidia-cudnn, nvidia-cufft, nvidia-cusolver, nvidia-nccl, triton,
...) before the `--no-deps` swap touches torch alone — those siblings are
left installed and unused. Separately, uv caches every downloaded wheel
under `/root/.cache/uv`, and that cache persists inside the Docker layer
regardless of what gets uninstalled afterward. Measured on a real image:
5.81GB total, of which the installed venv was only 882MB and the wheel
cache alone was 5.3GB. Fix at the time: explicitly uninstall the orphaned
`nvidia-*`/`triton` packages after the torch swap, and set
`ENV UV_NO_CACHE=1` before `uv sync`, so the cache never lands in a layer
at all. Got the image down to 1.09GB — **but this whole a/b/c/d chain was
still the wrong shape.** See the update below.

**Update (2026-07-12, same day): the swap-after-sync approach was
superseded entirely.** All four fixes above patched symptoms of one root
cause: `pyproject.toml`'s `[tool.uv.sources]` pins torch to the cu130
index for *any* Linux (`sys_platform == 'linux' or sys_platform ==
'win32'`), so a CPU build's `uv sync` always resolved and downloaded the
full CUDA dependency tree before anything could be swapped out — a
build-then-diet approach, not a CPU-only build. Even a genuinely 1.09GB
final image was still burning real time and bandwidth downloading ~5GB of
wheels it never kept, on every rebuild.

**Real fix:** three separate, self-contained uv projects instead of one
shared lockfile with post-hoc patching:
- root `pyproject.toml`/`uv.lock` — unchanged, local dev (still resolves
  torch from cu130, exactly as before this session started).
- `docker/cpu/pyproject.toml` + its own `uv.lock` — torch pinned
  unconditionally to `https://download.pytorch.org/whl/cpu`, no markers,
  no extras. `uv sync` for this project never touches the cu130 index —
  confirmed via a real `docker build --no-cache`: zero `cu130`/`nvidia-`/
  `triton` mentions anywhere in the build log.
- `docker/gpu/pyproject.toml` + its own `uv.lock` — torch pinned to cu130,
  same resolution as before, just isolated into its own project.

Both trainer Dockerfiles now `COPY docker/{cpu,gpu}/pyproject.toml
docker/{cpu,gpu}/uv.lock ./` instead of the root files. `Dockerfile.
trainer-cpu`'s entire swap/strip/uninstall block (a/b/c/d above) is gone —
replaced by a plain `RUN uv sync --no-dev`. `--no-sync` stays in both
Dockerfiles' `CMD` as a defensive habit (no longer fixing an active bug,
since each image's own lockfile now matches what's installed, but it still
skips the reconcile-against-lock check on every cold start).

**Tradeoff accepted knowingly:** the three `pyproject.toml` files
duplicate the non-torch dependency list (fastapi, uvicorn, numpy, ...) by
hand — normally a DRY violation this project avoids, but chosen
deliberately over uv's `[project.optional-dependencies]` +
`[tool.uv.conflicts]` mechanism (which can express this with one shared
lockfile) because that approach requires remembering to pass `--extra
cpu`/`--extra gpu` on every local `uv sync`/`uv run`, and a bare `uv sync`
without the right extra would silently uninstall torch from whichever venv
it's run against. Three independent project directories mean the correct
dependency set is just "whichever directory you're in" — no flag to
forget, and the root project (local dev) is completely untouched by any of
this, so there's zero risk to the existing local GPU dev venv. Keep the
three dependency lists in sync by hand when adding/changing a non-torch
package.

**General lesson (reinforced):** the a/b/c/d fixes were each individually
correct and each verified against a real build — but verifying that a
patch works is not the same as verifying the *architecture* is right.
"The final image is small and torch is +cpu" was true and still masked
"the build downloads 5GB of CUDA it doesn't need." When a fix requires
multiple layered patches to route around one shared piece of state (here,
one lockfile pinned for the wrong target), that's a signal to check
whether the shared state itself should be split, not patched further.

**General lesson:** none of these four would have been caught by reading
the Dockerfile — each needed an actual `docker build` + `docker run` to
surface (a resolver error, a runtime version check, a `du -sh` on the
running container). Don't trust a package manager's "reinstall just this
one thing" flag, a version string, a `CMD`'s idempotency, or a "the swap
worked" build log to mean what they imply — verify narrow-scope operations
are actually narrow, and check the artifact you actually shipped, before
relying on it somewhere as consequential as a production image build.

---

## 13. VM-Attached Service Accounts Need No Nebius Profile — and Docker Registry Auth Splits Root/User

**Found live on a fresh GPU VM (2026-07-12), diagnosing a `docker push`
failure step by step rather than guessing:**

**a) `nebius profile create` is the wrong tool for a VM-attached service
account.** The VM was created with `--service-account-id` pointing at
`mlflow-sa` (see other Nebius projects' `nebius_provision.sh` for that
flag), which is Nebius's actual mechanism for "VM-attached SA" — not
something `setup_gpu.sh` originally accounted for at all (it never
installed the `nebius` CLI in the first place). Running `nebius profile
create` interactively offered two paths — federation (browser OAuth,
which fails outright on a headless VM: it opens `http://127.0.0.1:<port>`,
unreachable from a local browser) or a manual service-account key file
(which was never generated for `mlflow-sa` and isn't needed). Confirmed via
Nebius's own docs (their in-console assistant, citing
"How to work with the Nebius AI Cloud CLI on a Compute virtual machine")
that an attached SA's token is read automatically from
`/mnt/cloud-metadata/token` — `nebius iam whoami` works immediately with
**no profile created at all**.

**b) `nebius registry configure-helper` splits across root/user in a way
that isn't obvious from the error message.** It does two things: installs
a `docker-credential-nebius` binary into `/usr/local/bin` (needs root to
write), and updates `~/.docker/config.json` to reference that helper
(inherently per-user — this is Docker's own config design, not a Nebius
quirk, and the same split exists with AWS ECR's/GCP's credential helpers).
Running the whole command under `sudo` to satisfy the binary write
silently wrote the config to `/root/.docker/config.json` instead of the
real user's — Nebius's own quickstart explicitly warns against this
("Do not run Docker commands as root. The credential helper is configured
for your user account, so root may not be able to access the
credentials."), but it's easy to miss until `docker push` fails for a
completely different-looking reason afterward.

**Fix:** `setup_gpu.sh` now (1) calls the existing
`scripts/install_nebius_cli.sh` after the repo clone (it already no-ops
gracefully when the manual-key env vars are absent — exactly the
VM-attached-SA case), (2) runs `sudo nebius registry configure-helper`
once for the root-only binary write, then (3) copies the resulting
`/root/.docker/config.json` into `$HOME/.docker/config.json` and fixes
ownership to the real user, rather than trying to avoid `sudo` entirely or
running docker itself as root.

**General lesson:** a permission error on one specific file
(`/usr/local/bin/docker-credential-nebius: permission denied`) looks like
it wants a blanket `sudo` on the whole command — but the fix that actually
works is scoping `sudo` to just the part of the operation that needs it
(the binary write) and keeping everything else (the docker daemon group
membership, the docker config file, actual `docker build`/`push` calls)
on the regular user. Mixing root and non-root across steps of one workflow
is what caused the confusion here, not the underlying design of either
Docker's or Nebius's credential mechanism — both are standard, shared
patterns across every major cloud registry.

---

## 14. Backgrounding `_start_remote_run` Exposed a Latent `execution_backend` Bug

**Context:** to let Stop cancel an in-flight provisioning run (§ Part F,
2026-07-12), `start_training()`'s remote path was changed from `await
_start_remote_run(...)` directly inside the HTTP request to
`asyncio.create_task(...)`, so `/start` returns immediately with the
`run_id` instead of blocking for up to ~6 minutes.

**What broke:** `training_runs.execution_backend` defaults to `'local'` in
the schema (`db.py`), and `_start_remote_run` only ever set it to
`'nebius_endpoint'` near the *end* of provisioning — after mirroring the
experiment onto the remote endpoint. Before backgrounding, this was
invisible: the request blocked until that write happened, so nothing ever
observed the DB in between. After backgrounding, `/start` returns
immediately while `execution_backend` is still at its `'local'` default,
and the frontend's normal status-polling loop can now see that
intermediate state for the *entire* provisioning window — a user who
picked CPU + Serverless would see the Experiments page show "local" until
provisioning finished minutes later.

**Fix:** pass `execution_backend=req.backend` into `create_training_run()`
at row-creation time, so it's correct from the very first read instead of
relying on a stale schema default plus a later correction.

**General lesson:** backgrounding a previously-synchronous call doesn't
just change timing — it can turn an intermediate DB/state value that used
to be unobservable (because nothing could read it before the blocking call
finished) into something a poller now sees directly. When making a
blocking call async/backgrounded, audit every field that call used to set
"eventually" and check whether something now reads it *before* that
point — don't assume "it always ends up correct" is the same as "it's
correct the whole time it's observable."

---

## 15. Chatbot Told a User Their Paused, Step-307 Run Was "Cancelled at Step 0"

**Context:** a user reported the Lab Assistant giving a confidently wrong
360° status assessment — "cancelled, step 0/1000, no metrics" — while they
were actively prompting a **paused run at step 307**. Two separate, real
bugs, found by tracing the actual grounding code rather than trusting the
chatbot's own (also wrong) self-diagnosis, which invented an unverified
"race condition in the status API" explanation.

**Bug 1 — wrong run selected.** `backend/api/chatbot.py`'s
`post_message()` grounds every turn in `list_runs_for_experiment(...)[0]`
as "the current run." That function sorted `ORDER BY started_at DESC` —
but `started_at` is nullable and only gets set once training actually
begins, not at row creation. An experiment with an older, already-terminal
run (whose `started_at` got set) and a newer, genuinely active run (whose
`started_at` timing didn't line up the same way) could return the *older*
run as `[0]`. `list_open_runs()` elsewhere in the same file already
correctly used `ORDER BY training_runs.id DESC` — an `AUTOINCREMENT`
primary key, always monotonic at creation time, never `NULL`. Fixed
`list_runs_for_experiment` to match.

**Bug 2 — lifecycle events buried by polling noise.** Even with the right
run, the chatbot's only other window into "what actually happened
recently" was `_get_log_tail()` — a fixed-size tail of *any* log category.
In practice that tail is dominated by `lab.request` polling lines (e.g.
`GET /api/nebius/workers/cpu` every few seconds) — a `PAUSE`/`STOP` event
could scroll out of a 50-line tail within a couple of minutes even though
it's the single most important fact about the run. Added
`_get_recent_training_events(run_id, n)`, filtering specifically on the
`lab.training` category + `run_id=<N>` — same pattern as the existing
`_get_recent_errors`/`_get_prompt_history` category-filtered scans, so
lifecycle events for the current run are guaranteed visible regardless of
how much other traffic happened in between.

**A precision gotcha hit while building the fix for Bug 2:**
`_get_prompt_history`/`_get_last_audit_change` use an `"id=<N> "` marker
with a trailing space as an implicit word-boundary check. Several
`lab.training` messages (`STOP run_id=1204`, `PAUSE requested
run_id=1204`) end right at the digits with *nothing* after — a
trailing-space marker would have silently matched zero of exactly the
lines that matter most. Fixed by matching the bare `run_id=<N>` substring
and explicitly checking the next character isn't a digit (so `run_id=5`
can't false-match inside a line about `run_id=50`), rather than relying on
trailing content that isn't always there.

**General lesson:** when a report says "the chatbot/system is confidently
wrong," don't accept its own explanation for why — trace the actual data
path it reads from (here: two independent gaps in *what the chatbot could
see*, not a hallucination or a "race condition"). Both were found and
fixed by reading the real query and the real log line formats, not by
guessing at plausible-sounding causes.

---

## 16. Open Runs Showed a Successfully-Running Remote Run as Permanently QUEUED

**Context:** found while investigating §15 — a user reported the Open Runs
page showing run #139 as `QUEUED`/step 0 while it was actually training
successfully. Confirmed by querying the row directly: `execution_backend`,
`remote_endpoint_id`, and `remote_run_id` were all correctly set (proof
`_start_remote_run` completed successfully), but `status` was still
`queued` — its value from row creation, never touched again.

**Root cause:** `_start_remote_run`'s final DB write only ever set
`execution_backend`/`remote_endpoint_id`/`remote_run_id` — never `status`.
For a remote run, the *live* status genuinely lives on the remote
endpoint's own DB from that point on; the local row was never meant to be
kept in sync with it turn-by-turn. `run_status()` (what the Experiment
page polls) already knew this and proxies to the remote endpoint for live
status when `_is_remote(db_run)`. `list_open_runs()` didn't — it read the
local `status` column directly, which for any remote run past handoff was
permanently frozen. A second consequence of the same gap:
`db.list_open_runs()`'s own terminal-status filter also only sees that
stale local value, so a remote run that's genuinely `completed`/`failed`
would never have dropped out of the Open Runs list either.

**Fix, two parts:**
1. `list_open_runs()` (the route) now proxies to each remote run's own
   endpoint for live `status`/`current_step`/`total_steps` before
   returning the list — same `_proxy()` helper `run_status()` already
   uses. Falls back to the stale local value on a proxy failure (logged,
   not raised) rather than breaking the whole list over one unreachable
   endpoint. Re-filters on the *live* status afterward, so a
   genuinely-terminal remote run correctly disappears from the list too.
2. `_start_remote_run` now also sets `status=RunStatus.RUNNING` in that
   same final write — a reasonable baseline for any other code path that
   might read the local row directly without proxying, even though it
   won't itself track later remote transitions (e.g. a later remote
   pause). The live overlay in (1) is what makes Open Runs itself
   correct; this is a complementary "don't leave a misleading local
   value lying around" fix, not a full sync mechanism.

**General lesson:** when one field only gets updated by a narrow code
path (here: local `status`, only advanced by the local subprocess runner
or by proxied writes triggered through specific routes), a *different*
consumer of that same table can silently see a stale value forever if it
reads the column directly instead of going through whatever keeps it
fresh. Grep for every reader of a column before trusting that "it gets
updated somewhere" means "it gets updated everywhere that reads it."

---

## 17. Idle-Timeout Clock Never Reset by Actual User Engagement (Pause/Resume/Prompt)

**Context:** a user saw an idle-timeout warning banner ("stops in ~3 min")
on a CPU worker despite having been actively engaged the whole time —
training for ~9 minutes, then pausing and prompting the paused model for
up to ~10 more minutes. They correctly suspected the idle clock wasn't
using the right signal, and separately asked whether the backend and
frontend might be comparing timestamps across different timezones (BST
vs UTC) — a reasonable, worth-checking hypothesis that turned out **not**
to be the cause: `idle_monitor.py::seconds_since()` already explicitly
treats the stored value as UTC and compares against UTC `now()`, and the
frontend (`WorkerIdleBanner.tsx`) does pure `idle_timeout_seconds -
seconds_idle` number arithmetic with no date parsing at all — both
checked directly, neither had a timezone bug. (One dead end worth noting
honestly: an earlier manual check in this same investigation compared a
UTC DB timestamp against a local-BST `date` command output without
converting, wrongly concluding the worker had been idle for over an hour —
caught and corrected before it went anywhere, but a good example of
exactly the mistake being investigated, made by the investigator.)

**Real root cause:** `db.touch_worker_session()` (resets
`last_activity_at`) was only ever called from three places: worker
acquisition/reuse (`ensure_worker`), once at the end of a successful
`_start_remote_run`, and the manual "Continue session" heartbeat button.
`pause_training`/`resume_training`/`prompt_model`'s remote branches all
proxy real, active work to the endpoint (prompting a paused model
literally runs an inference forward pass there) but never touched the
clock. A run that's paused-and-prompted for long enough looks
indistinguishable from a worker nobody has touched since it started.

**Fix:** added `_touch_worker_for_run(db_run)`, called after
pause/resume/prompt's remote proxy calls succeed. Deliberately **not**
added inside `_proxy()` itself, which would also cover passive
`/status`/`/metrics` polling (happens automatically on a timer regardless
of whether anyone's actually engaged) — touching on that too would make
idle-timeout effectively never fire as long as a browser tab is left
open, defeating its purpose entirely. The distinction that matters is
explicit user action vs. automatic background polling, not "did a request
happen."

**Follow-up, same day — broadened per explicit user request:**

1. `run_status()`'s remote branch now syncs the local row's `status`/
   `current_step`/`total_steps` to the live-fetched value on *any*
   detected transition, not just a special-cased "completed" check —
   paused, cancelled, and failed all matter equally, not just successful
   completion. Touches the worker specifically when the transition is
   *into* a terminal state (`completed`/`failed`/`cancelled`) that wasn't
   already terminal locally — "the run just finished on its own" is a
   legitimate activity signal even without an explicit local action.
2. Added `frontend/src/hooks/useActivityHeartbeat.ts` — a throttled
   (max once per 60s) global listener on scroll/mousemove/keydown/click,
   calling the existing heartbeat endpoint (the one "Continue session"
   already used). This was an explicit, deliberate choice by the user
   over my own recommendation: I flagged that passive mouse/scroll
   activity is a weak signal for a *cost-control* timeout specifically
   (unlike a security session-timeout) — a tab left open with incidental
   scrolling could never idle out, and this app already has a purpose-built
   mechanism for exactly that ambiguous case (the explicit
   "Continue session" click). The user considered that tradeoff and chose
   to include scroll/mouse tracking anyway, prioritizing convenience over
   guaranteed timeout enforcement. Recorded here so the tradeoff isn't
   rediscovered as a surprise later — it was a conscious choice, not an
   oversight.

**General lesson:** "idle" needs a definition, and it's easy to
implement a definition that's too narrow (only counts session start) or
too broad (counts a polling timer). Both are equally wrong for different
reasons. When adding an activity-tracking signal, ask what specifically
should and shouldn't count *before* picking where to wire it in — not
just "does this request touch the resource."

## §18: Notes moved from experiment-scoped to run-scoped

**Problem:** notes lived on `experiments`, not `training_runs`. A note
written about one run (e.g. "loss spiked around step 200, probably the LR
warmup") stays attached to the experiment forever, even after the config
changes and a new run starts. The note is now misleading — it describes a
run that no longer matches the current config — but nothing ever clears
it.

**Fix:** `notes_md` added to `training_runs` (schema + migration in
`backend/db.py`), new `GET`/`PATCH /api/training/{run_id}/notes` routes
(`backend/api/training.py`), reusing the existing generic
`db.get_training_run`/`db.update_training_run` — no new db.py functions
needed. `ExperimentNotes.tsx` now takes `runId` instead of `experimentId`;
textarea is disabled with a placeholder when `runId == null` (no run
started yet). The "wipe on new run" behavior needed no explicit code: since
notes are a column on the per-run row and that column defaults to `''`,
every new `training_runs` row starts blank automatically — there's no
copy-forward step to get wrong. The old experiment-level `notes_md`
column/route in `backend/api/experiments.py` was left in place, unused by
the frontend now, rather than deleted — not worth the churn for a POC.

## §19: Four fixes from a real GPU incident (2026-07-12)

A live GPU run surfaced several compounding gaps at once — a stopped
endpoint got abandoned instead of restarted, the frontend showed
"disconnected" for four minutes of totally normal provisioning, the
chatbot reported an empty loss trend for a run that actually completed,
and the chatbot fabricated "I'm checking with the engineering team" when
asked about it. Fixed independently, documented together since they were
found together:

1. **Stopped endpoint abandoned instead of restarted.** `create_new_worker()`
   (`backend/nebius/worker_manager.py`) only ever checked for a live
   RUNNING endpoint to adopt (`find_running_endpoint`) — never a STOPPED
   one it could cheaply restart. Generalized
   `endpoints_client.find_running_endpoint` into `find_endpoint(name,
   state)`, and `create_new_worker` now checks RUNNING → STOPPED (restart
   via `start_endpoint`) → create fresh, in that order. Lives in the shared
   kernel, so both CPU and GPU get it, and both the running app and
   `scripts/create_nebius_endpoint.py` benefit.
2. **"Disconnected" banner during totally normal provisioning.** `_proxy()`
   builds the remote request path from `db_run["remote_run_id"]`, which
   stays `None` until `_start_remote_run`'s background task finishes
   mirroring the run — which can take several minutes for a cold GPU.
   Every poll during that window built a URL with `None` in it and failed,
   indistinguishable from a real outage. Fix: `run_status()` now checks
   `remote_run_id is None` first and serves the local (already `QUEUED`)
   status instead of proxying — no new status value needed, and no
   frontend changes, since `QUEUED` was already a normal status the UI
   handles.
3. **Chatbot always saw `train_loss_history: []` for remote runs.**
   Confirmed via a real transcript: a completed GPU run's loss trend was
   empty even though training genuinely finished. Root cause:
   `train_loss_history` is written in exactly one place —
   `train_worker.py::write_metric()` — which only ever runs inside the
   *local* training loop. A remote run's metrics live only on the
   Nebius container's own disk; nothing copied them back. Fix:
   `get_metrics()` (`backend/api/training.py`) now mirrors proxied remote
   metrics into the local row's `train_loss_history`/`val_loss_history`
   columns on every poll, piggybacking on the request the frontend already
   fires every ~2s rather than adding a new one.
4. **Chatbot fabricated follow-up action.** The same transcript included
   "I'm checking with the engineering team about fixes" — invented; the
   chatbot has no such capability and no such process exists. It also
   guessed the bug lived in `backend/training/runner.py` (wrong file)
   despite its own system prompt saying to admit lack of visibility rather
   than guess. Added an explicit line to `_SYSTEM_PROMPT`
   (`backend/chatbot/context.py`) forbidding claims of taking action
   outside the conversation.

**Follow-up, same day:** fix #2 above only patched `run_status()` —
`get_metrics()` has the exact same `remote_run_id is None` proxy bug, and
the frontend's poll loop (`App.tsx::pollStatus`) calls both endpoints back
to back, treating *either* one failing as a disconnect. So the red
"backend disconnected" banner kept appearing during totally normal
provisioning even after status correctly reported `queued` — the two
signals were contradicting each other on screen. Applied the identical
short-circuit to `get_metrics()` (return `[]` while `remote_run_id` is
`None` — there genuinely are no metrics yet). Also added a dedicated blue
"waiting for the serverless endpoint to start, up to ~5 min for a cold
GPU" banner shown specifically when `status === "queued"` on a remote run,
so a legitimately-slow cold start reads as "this is expected, hang on"
instead of looking identical to a real outage.

**Second follow-up, same day — the actual root cause of fix #1's remaining
gap:** deep log forensics (grepping every session log for one specific run)
showed `ensure_worker()`'s own `start_endpoint()` call timing out at 180s
while the GPU endpoint was still genuinely, successfully starting up on
Nebius's side. The code assumed a start timeout meant "deleted outside the
app" and abandoned the endpoint, creating a wasteful duplicate — the run
that triggered this was left `failed` even though the endpoint it was
waiting on came up fine on its own shortly after. Two changes:
`nebius_endpoint_start_timeout_seconds` raised from 180s to 300s
(`config/settings.py`), and — more importantly —
`ensure_worker()` (`backend/nebius/worker_manager.py`) now calls
`get_endpoint()` on a start timeout *before* assuming deletion: if the
endpoint still exists in any state, it keeps waiting on it; only falls
back to creating fresh if `get_endpoint` itself confirms the endpoint is
genuinely gone. A client-side timeout is a fact about *our* patience, not
about whether the remote resource still exists — conflating the two was
the actual bug, not just the timeout being too short.

## §20: Lab Assistant UX — typing indicator, feedback, response length

Three small requested changes, one real bug found while building the
second one:

- **Typing indicator**: the assistant message bubble is added to state with
  empty content the moment streaming starts (`useChatStream.ts`), before
  any delta arrives. `ChatPanel.tsx` renders three CSS-animated dots
  (`.typing-dots` in `index.css`) whenever an assistant message's content
  is still `""`, and swaps to the real text automatically once the first
  token lands — no extra state needed, the emptiness itself is the signal.
- **Thumbs up/down**: new `feedback` column on `chat_messages`
  (`backend/db.py`), `PATCH /api/chatbot/messages/{message_id}/feedback`.
  Building this surfaced a real bug: assistant messages get a client-side
  negative placeholder id the instant streaming starts
  (`useChatStream.ts::localMessage`), and that id was never reconciled
  with the real DB row id once `add_chat_message()` persisted it —
  harmless for the copy button (never left the browser) but would have
  made every feedback PATCH 404 against an id the server never assigned.
  Fixed by having the `done` SSE event include the real `message_id`,
  which the client swaps in for the placeholder.
- **Response length**: `_SYSTEM_PROMPT` (`backend/chatbot/context.py`) now
  asks for ~300 words. Deliberately a soft prompt instruction, not a
  backend truncation — cutting a response off at N characters would chop
  it mid-sentence, which is worse than an occasionally-long answer.

---

## File Layout

See `README.md` for project structure and setup instructions.
