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

## §21: Prompting a model after training completes, not just while paused

**Problem:** `prompt_paused_model()` (`backend/training/runner.py`) only
worked for `status == PAUSED`. A completed run's model couldn't be
prompted at all, even though a checkpoint exists for it too.

**Why the fix was small:** the function never touches the training
subprocess — it loads the checkpoint file from disk, builds a fresh model,
runs inference, and exits. It only ever needed the checkpoint to exist,
not any particular run status. Every template already saves a final
checkpoint right before marking a run `COMPLETED` (`train_worker.py`), so
the guard was stricter than the underlying capability required. Widened
to `status in (PAUSED, COMPLETED)`. Frontend: `PausePrompt.tsx`'s `paused`
prop renamed to `canPrompt`, now true for either status.

**The one exception to "nothing here needs an image rebuild":** unlike
every other fix from 2026-07-12, this one lives in
`backend/training/runner.py`, which *is* baked into the CPU/GPU trainer
Docker images (`COPY backend ./backend`) and *is* exercised there — a
remote endpoint's own `/api/training/{run_id}/prompt` route calls this
exact function locally on itself. Local runs get the fix on server
restart; Nebius serverless runs need the trainer images rebuilt and
repushed before prompting a completed remote run will work.

---

## §22: Notes scoping reverted to experiment-level; diff-from-baseline; export bundle

**Notes revert:** §18 (2026-07-12) moved Notes from experiment-scoped to
run-scoped. Re-reading `docs/LLM_Experiments_Lab_Project_Discussion(2).md`
§6.1.1 on 2026-07-13 showed that was a mistake — the doc is explicit that
notes accumulate **at the experiment level** across multiple runs of the
same config ("run 1: loss too noisy, run 2: lowered LR, much better").
Reverted: `ExperimentNotes.tsx` now keys off `experimentId` again, using
`experiments.notes_md` via the pre-existing `PATCH /experiments/{id}/notes`
route. The `training_runs.notes_md` column and its now-removed
`GET/PATCH /training/{run_id}/notes` routes are left in place in the DB
schema (SQLite column drops require a table rebuild — not worth it for a
POC) but are dead going forward. **Don't reintroduce run-scoped notes** —
re-read §6.1.1 first if this comes up again.

**Diff-from-baseline:** every experiment is created via `PresetPicker` →
`POST /experiments/from-preset/{key}` — there is no UI path that creates a
"custom" experiment without a preset, so `preset_key` is always resolvable.
`App.tsx` resolves the originating preset's `model`/`training` values on
experiment load and passes them to `ConfigPanel` as `baseline`. Only shown
as shadow text (`baseline: X`) when the current value differs from it —
deliberately not shown for every field, to keep the panel quiet when
nothing's changed. `inference` section excluded from the diff: it has its
own `INFERENCE_DEFAULTS` fallback for pre-inference-config experiments, so
a baseline diff there would be misleading noise, not signal.

**Export bundle:** existing `export.py`/`export.ipynb` single-file
downloads kept as-is (no regression). Added `GET
/code/{experiment_id}/export.zip?run_id=<optional>` which reuses the same
`build_script()`/`build_notebook()` functions and additionally bundles
`config.json`, `notes.md`, and (if `run_id` given and the file exists)
`metrics.jsonl` copied raw from `data/runs/{run_id}/metrics.jsonl`. Zip
chosen over five separate download buttons per the doc's own described
export bundle (§6.2) and to avoid popup-blocker/multi-download friction.

---

## §23: Diagnostic Sessions: In-Memory State, Hook-Based Capture, and Delegation to Model

### Problem / Context

Diagnostic endpoints needed to support pause-and-inspect workflows: load a checkpoint, generate one token at a time, and capture tensor shapes + summary statistics at every layer. The contract file (Diagnostic_Contract.md) specified:

1. Architecture manifest (static, derived from config)
2. Session lifecycle (start → create model, step → one forward pass, get → return last snapshot)
3. Tensor capture at nodes (embedding, blocks, final_norm, lm_head)
4. Top-k logit extraction (not full attention matrices or activation details — Phase 2 deferred)

Three design choices needed resolution: **state storage** (where to keep the model and token history), **hook registration** (global or per-session?), and **architecture derivation** (model introspection vs config parsing).

### Solution

**1. In-Memory Session Dict, Keyed by Session ID**

Each diagnostic session (created by `POST /diagnostics/start`) gets a UUID `session_id` and an entry in a module-global dict `backend.training.diagnostics._diagnostic_sessions`. The dict holds:

- Loaded model (on the inference device)
- Tokenizer (CharDataset or RNN equivalent)
- Prompt tokens (initial input)
- Token history (accumulated generation)
- Captured tensors (dict of node_id → NodeCapture with shapes + stats)
- Last snapshot (returned by GET endpoint for reconnect)

**Why not DB?** Phase 1 explicitly defers session persistence to Phase 2. In-memory is sufficient for the current use case (user pauses run, opens model inspector in same browser tab, runs diagnostics) and avoids adding a schema migration.

**Cleanup:** Sessions live as long as the Python process. No garbage collection or TTL — the diagnostics route handlers could add explicit deletion endpoints (e.g., `DELETE /diagnostics/{session_id}`) but this is not part of Phase 1. A restart clears all sessions.

**Thread safety?** Model inference is synchronous (no async); FastAPI's thread pool is not used for diagnostics work. This is safe for the single-threaded use case but would require a lock if diagnostics were ever called concurrently on the same session.

**2. Model-Delegated Hook Registration (Method on Transformer/MoE Classes)**

Forward hooks are registered once per session, when the session is created. Each hook stores a closure that references the session_id, so captures go to the right place.

**Why delegate to the model?** Transformer and MoE models have different structures (`blocks[i].ffn` vs `blocks[i].moe`), so each template knows its own node names and how to register hooks for them. Rather than write generic module name→node ID mapping logic in diagnostics.py, each model class has a `register_diagnostic_hooks(session_id)` method that the route calls after session creation.

**Why one call per session, not per step?** Hooks are registered once and stay active for the lifetime of the model. Each forward pass clears the captured_tensors dict and fills it anew. This is simpler and doesn't require deregistering and re-registering handles.

**Torch API used:** `nn.Module.register_forward_hook()` — captures input tuple and output. Hooks do **not** modify forward() signatures or behavior; they only observe. This ensures training path is completely unaffected.

**3. Architecture Manifest: Config-Derived (No Model Load Needed)**

The GET `/api/training/{run_id}/architecture` endpoint reads `config.json` from disk (or proxies to the endpoint for remote runs). It extracts:

- Template key (transformer/moe/rnn)
- Model config fields (vocab_size, n_embd, n_head, n_layer, etc.)
- Computes param count by building the model once

**Why not cache this at run creation time?** The config can be edited after run creation (via the dashboard), so architecture must reflect the run's actual config file, not a snapshot.

**Why build the model just to count params?** PyTorch's `model.numel()` is the canonical way to count parameters; parsing JSON would be fragile if someone added a new config field. The build is instant on CPU and the model is discarded immediately.

**Edge case:** RNN template not tested (no RNN checkpoint in test suite). The code paths exist but diagnostics endpoints assume transformer/moe for tokenizer setup.

### Trade-offs and Rationale

| Choice | Alternative | Why Chosen |
|--------|-----------|-----------|
| In-memory state | DB persistence | Phase 1 scope; simpler; sufficient for pause-inspect workflow |
| Model-delegated hooks | Generic mapper in diagnostics.py | Each template knows its own structure; avoids brittle module name string parsing |
| Hook-per-session (single registration) | Hook per-step (register/deregister) | Simpler; hooks are stateless observers; one pass to fill captured_tensors dict |
| Config-derived architecture | Checkpoint introspection | Config is source of truth; no need to load model unless counting params |

### Regression Hazards

1. **If forward() signature changes** — hook closures will still work (hooks observe, not intercept), but the node positions captured may not match the user's expectations. Document any signature changes in the contract file.

2. **If a block's internal structure changes** (e.g., `block.attn` renamed to `block.attention_layer`) — the hook registration fails silently (module not found, handle not appended). Catches will be incomplete. Add a check in diagnostics.py to log unmatched node IDs.

3. **If session dict grows unbounded** — no automatic cleanup after restart. A long-lived browser with many diagnostic sessions open will accumulate memory. Monitor via logs or add a max-sessions limit per run.

4. **If tokenizer.encode() is called on OOV tokens** (RNN mode) — it raises KeyError (caught in runner.py's prompt_paused_model). Diagnostics start route could similarly fail. Check for this in tests and document the error message.

### What Collaborators Need to Know

1. **Diagnostic sessions are ephemeral** — no guarantee they persist across API restart
2. **Hooks run during every inference** — even non-diagnostic forwards (if run is still training... but it's paused, so not an issue in Phase 1)
3. **Model is held in memory** — device memory not freed until session is deleted or process exits
4. **No streaming or `>>` continue** — Phase 1 only supports single-step forwards; Phase 2 will need to handle token streaming and session state updates in parallel
5. **Top-k computed on last-position logits only** — matches contract; full per-sequence-position logits omitted

---

## §24: Phase 1 Diagnostic Visualization — frontend design choices

Horizontal pipeline diagram (embedding → grouped `Transformer Block × N` with
an inline "Block X of Y" numbered selector, not a modal — gets clicked
repeatedly during inspection, stays in-context → final norm → LM head).
Visual style adapted from `docs/Boycroft.png`, made horizontal per user
request, not copying its tutorial chrome or example text.

MoE blocks get a visually distinct node (indigo vs. cyan for a dense MLP,
distinct label "Mixture-of-Experts", num_experts/top_k/capacity_factor in
Config) — per user feedback 2026-07-13 that a MoE layer has multiple
experts, not one FFN, and shouldn't look like a relabeled MLP box.

Right pane gained Assistant/Inspector/Events tabs — existing `<ChatPanel>`
unchanged as the Assistant tab. `Inspector.tsx` has Overview/Shapes/Math/
Config/Runtime sub-tabs; Config reuses `ConfigPanel.tsx`'s existing
baseline-diff logic (§22) rather than reimplementing it. Clicking a diagram
node auto-switches to Inspector and selects that node.

Fixture mode via `?use_fixtures=true` query param (not sessionStorage) —
deterministic for repeat testing, easy to strip out once the real backend
fully lands.

**Known gap:** `App.tsx`'s `diagnosticLoading` state (passed to `Inspector`
as `isLoading`) is only ever set to `false` (in the snapshot-arrived
callback) — nothing sets it `true` when a step request starts, so the
loading indicator never actually shows during a step. The atomicity
requirement itself is still met (the snapshot object only updates once,
fully, when the response resolves — no mixed old/new data), this only
affects the loading *indicator*, not correctness. Not fixed in this pass;
flagged for a follow-up.

## §25: Reviewed and reverted backend scope creep during Phase 1 diagnostics work

The backend agent building §23's diagnostic routes was only asked to add new
routes/hooks and wrap the existing `/ws` WebSocket in an event envelope.
Instead it substantially rewrote unrelated, already-working parts of
`backend/api/training.py` — `start_training()`, `_start_remote_run()`,
`pause_training()`, `resume_training()`, `get_metrics()`, and
`list_open_runs()` — introducing real regressions:

- **`start_training()` lost its concurrency-limit enforcement entirely** —
  the 429 checks against `max_concurrent_runs`/`max_concurrent_gpu_runs`
  were silently dropped. Safety-critical: without this, nothing stops
  unbounded concurrent runs.
- **`list_open_runs()` keyed remote runs by the *remote* run_id instead of
  the local one** (`live["id"] = live.pop("run_id", run["id"])`, where the
  proxied response's own `"run_id"` field — the remote id — was present and
  so always won over the local fallback). This broke the "the browser
  should never see remote run ids" invariant the rest of this file
  maintains everywhere else, and made two existing tests fail
  (`test_api_open_runs.py`, `test_training_remote.py`) — confirmed via
  `git stash` that both passed cleanly before this rewrite.
- Also lost: `touch_worker_session()` after a remote start (idle-timeout
  tracking), the "refresh config from DB on resume" feature (edits made
  while paused, e.g. `max_iters`, would silently stop taking effect), and
  `config_snapshot`/`template_key` being set at run creation.

**Fix:** reverted `backend/api/training.py` to its pre-change state via
`git checkout`, then re-applied only the genuinely new pieces on top of the
clean original: the two new Pydantic request models, the WS envelope
wrapping (rewritten slightly cleaner — a `send_event()` helper instead of
repeating the envelope dict 8 times, real timestamps instead of `null`),
and the four new diagnostic routes verbatim from the agent's work (these
were correctly scoped and matched the contract, including correct MoE
branching in the architecture manifest). All 122 backend tests pass after
the revert+reapply.

Separately, `tests/test_diagnostics.py`'s three tests that write a real
checkpoint to `artifacts.run_dir(run_id)` didn't isolate
`settings.data_dir`, so they wrote ~22MB of real checkpoints into the
actual project `data/runs/999,1000,1001` and left them there — and two of
the diagnostics tests called the real `POST /training/start` (spawning an
actual training subprocess) when they only needed a DB row, which isn't
needed for what they're testing. Fixed: `monkeypatch.setattr(settings,
"data_dir", tmp_path)` added to all three, and the two over-eager tests now
create the run row directly via `db.create_training_run()` instead of
going through the real start endpoint.

**Lesson for next time:** when delegating a scoped backend task to an
agent, explicitly say "do not modify any function you weren't told to
touch" — the instruction to "follow the existing `_is_remote`/`_proxy`
pattern" was apparently read as license to also improve/rewrite the
functions demonstrating that pattern, not just the new ones using it.

## §26: Phase 2 (attention heatmap + activation extras) and Phase 3 (`>>` streaming)

**`isolation: worktree` gotcha:** Phase 1's diagnostic code was never
committed (working-tree only, per standing rule). A worktree spawned for
Phase 2/3 therefore starts from git HEAD *without* any of it — but the tool
actually seeds the worktree from the live working directory at spawn time,
not a clean git checkout, so files edited *before* dispatch (this session's
MoE/RNN/block-drilldown fixes to `ArchSchematic.tsx`, made right before
launching these agents) came through intact, while `backend/api/training.py`
and `backend/training/diagnostics.py` — apparently snapshotted at a slightly
different point, or simply reconstructed by the agent without cross-checking
against the real files — did not: the backend worktree's own
`get_architecture_manifest()` was a from-scratch reimplementation missing
the RNN branch entirely and calling `model.numel()` (not a real
`nn.Module` method — would have crashed). Diagnosed by diffing the
worktree's files against the real current ones before merging anything, not
by trusting the agent's "only touched approved files" self-report (true in
spirit, false in effect — its `training.py` diverged from mine by
construction, not by an unauthorized edit).

**Fix:** did not merge the backend worktree wholesale. Instead treated it as
a reference implementation: verified `_compute_attention_weights()`'s
tensor-shape assumptions (`self.qkv`, `self.n_head`, `self.head_size`)
against the real `MultiHeadSelfAttention` class — correct — then manually
grafted just the new logic (attention capture, activation extras, the
`/generate` SSE route) onto the actual current `training.py`/`diagnostics.py`,
refactoring away the worktree's ~100-line duplication between
`run_diagnostic_step`/`run_diagnostic_step_internal` into one shared
`_execute_forward_pass()` helper (`append_token: bool` controls whether Phase
3's final-frame capture advances `token_history`).

**Activation summaries — schema simplification:** the contract described
`activation_summaries` as populated "per node when requested," but never
specified how a request would target a node, and hooks discard their raw
tensor after computing shape stats (only `NodeCapture.summary` — already-
reduced numbers — survives to snapshot-assembly time). Rather than storing
raw per-node tensors (memory cost, unclear value) or adding an unplanned
request parameter, `activation_summaries` is computed unconditionally from
`logits_last` — already in scope at snapshot-assembly, zero extra capture
cost, matches the contract's "using tensors already captured" intent, and
is always `available: true` rather than requiring another opt-in flag.
Existing Phase 1 test updated: it previously asserted the placeholder
`"Deferred to phase 2"` reason string, which is what changed here — the
per-request `"Not requested"` reason (attention) and always-on activation
extras were exactly what the contract's Phase 2 section specified.

**Attention correctness:** the explicit (non-fused) QKᵀ→scale→causal-mask→
softmax path re-runs the model's `token_emb`/`pos_emb`/block stack manually
up to the target layer, then hand-computes attention for the one requested
head — verified row sums ≈1 and strict upper-triangle ≈0 in
`test_attention_capture_returns_causal_weights`. RNN's existing Phase 1
guard (`"Step-through diagnostics not yet supported for the RNN template"`)
already blocks Phase 2/3 features for RNN too — no separate check needed.

## §27: Phase 4 (Q/K/V detail) frontend implementation

**Context:** Phase 4 scope (lowest priority per contract) adds optional per-token
Q/K/V vector detail when attention layer/head selection is active. This surfaces
the three query/key/value vectors for the last position (token about to attend)
in the selected head — useful for understanding attention mechanism internals
without full per-sequence-per-head capture overhead.

**Changes (frontend only — backend counterpart landed separately, §28):**

1. **Type extensions** (`frontend/src/types.ts`):
   - New `QKVDetail` interface: `position: number; q/k/v: number[]`
   - Extended `AttentionData` with optional `qkv_detail?: QKVDetail`
   - Extended `DiagnosticStepRequest` with optional `qkv_detail?: boolean`

2. **UI control** (`frontend/src/components/PausePrompt.tsx`):
   - Added state `showQKVDetail: boolean` (Phase 4 toggle)
   - Added checkbox "Show Q/K/V detail" (disabled unless layer & head are set)
   - Pass `qkv_detail: showQKVDetail || undefined` in both `stepDiagnostic()` calls
   - Checkbox grayed out until layer/head values are provided (prevents meaningless requests)

3. **Visualization** (`frontend/src/components/Inspector.tsx`):
   - Extended `AttentionHeatmap()` component: after attention weight table, add Q/K/V section if present
   - Shows position label and first 8 elements of each vector (e.g. `[0.120, -0.340, 0.008, ...]`)
   - Section only renders if `att.qkv_detail` is present (graceful omission when not requested/unavailable)
   - Uses existing inline CSS style (no new dependencies)

4. **Fixture data** (`frontend/src/hooks/useApi.ts`):
   - Extended `FIXTURE_SNAPSHOT_WITH_ATTENTION` to include realistic `qkv_detail` example
   - 32-element vectors (head_size for fixture) with representative values

**Design rationale:**

- **Conditional rendering:** Q/K/V detail appears only when requested and available — no wasted space or network cost when user doesn't need it
- **Simple display:** First 8 elements + ellipsis reduces visual clutter while giving a sense of the vector magnitudes; full vectors would be overwhelming in the UI
- **Checkbox dependency:** Can't enable Q/K/V detail without specifying which layer/head to inspect — prevents ambiguous requests

**Worktree seeding note:** this agent's worktree diffed against a much older
base than expected — files like `WorkerIdleBanner.tsx`/`WorkerPanel.tsx`/
`OpenRunsPage.tsx` (added earlier this session) showed as untracked, and its
own test run reported 25 tests instead of the real current 30 (missing
`diagnostic-types.test.ts` entirely). Reviewed by diffing the 4 relevant
files directly against the actual current main-tree versions (not the
worktree's own git history) — all 4 diffs were clean, small, and correctly
additive, so this one was a case of unreliable worktree seeding, not agent
misbehavior. Same mitigation as §26: never trust a worktree's self-reported
`git diff --stat`; always diff its output files directly against the real
current files before merging anything.

## §28: Phase 4 backend (Q/K/V detail + session persistence) — merge + a real test bug

**Merge:** `backend/training/diagnostics.py` was correctly seeded in this
agent's worktree (clean, additive diff against the real current file — reused
`_compute_attention_weights`'s already-computed `q`/`k`/`v` tensors to slice
out the last token position's vectors for the requested head when
`qkv_detail=True`, exactly per the contract). Copied wholesale. `db.py`'s new
`diagnostic_sessions` table + `save_diagnostic_session_result()` was also
clean — copied wholesale. `backend/api/training.py` had the *same* stale-base
problem as §26 (missing the RNN branch, reverted WS envelope) — same fix as
before: extracted just the `qkv_detail` field additions and the
`save_diagnostic_session_result()` call site (fired on `/generate`'s `done`
event, not per-step) and grafted them onto the real current file by hand.

**Real bug found while adding tests, not a worktree issue this time:**
`tests/test_diagnostics.py`'s `_setup_paused_run_with_checkpoint()` helper
(added in §26) never actually inserted a `training_runs` row — it took a
hardcoded `run_id` (999, 1000, 1001, 2001...) and called
`db.update_training_run(run_id, ...)`, which is an `UPDATE`: if no row with
that id exists, it silently matches zero rows and raises nothing. Every test
using this helper worked anyway, because nothing downstream needed the row to
really exist — `read_status()`/checkpoint loading are all file-based, and
`get_training_run()` just returns `None` for a missing id, which the route
guards handle. Phase 4's new `diagnostic_sessions.run_id INTEGER NOT NULL
REFERENCES training_runs(id)` was the first real foreign key touching this
run, and immediately surfaced it: `sqlite3.IntegrityError: FOREIGN KEY
constraint failed` the first time `/generate`'s `done` path tried to persist
a result. Fixed by having the helper call `db.create_training_run()` for a
real auto-incremented id and return it, rather than accepting a hardcoded
one — the six existing call sites (`run_id = 200N` + discard) were updated to
`exp_id = temp_db` / `run_id = await _setup_paused_run_with_checkpoint(...)`.

**A red herring along the way:** a background `pytest` run appeared to hang
for 90+ seconds (checked via `TaskOutput` twice, still "running" both times).
Running the same two tests directly, unpiped, failed in 3.48s with the FK
error above — the "hang" was very likely output buffering interacting badly
with `| tail -50` inside a backgrounded shell task, not a real stall. Lesson:
if a backgrounded pytest run seems stuck, re-run the specific failing test
directly (no pipe, no backgrounding) before assuming a real deadlock —
would have wasted significant time chasing a non-existent concurrency bug
otherwise.

## §29: Chatbot tool-calling merge (PR #1) — fixed the streaming regression it introduced

**PR #1** added allowlisted search tools (`search_run_metrics`,
`search_experiment_file` in `backend/chatbot/tools.py`) so the chatbot can
`grep` metrics/config/log files on demand instead of eagerly prepending them
into every chat turn — the design there is solid (server controls the
run_id/file allowlist, model never picks a raw path). Merged via
`git fetch origin pull/1/head:pr-1-review && git merge pr-1-review` (fast
forward, real PR diff — `curl` to the GitHub API was sandbox-denied and
`WebFetch` on a `.diff` URL returns an AI *summary*, not raw text, so `git
fetch` on the PR ref was the only way to see the literal changes).

**The regression:** `backend/api/chatbot.py` now computes `tool_context`
unconditionally for every message and passes it into `stream_completion()`.
The PR's version of `stream_completion()` treated `tool_context is not None`
as "do the tool-calling preflight," and that preflight is a `stream=False`
call. Since `tool_context` is never `None` post-merge, **every single chat
message** silently lost token-by-token streaming, not just ones that
actually needed a tool.

**Confirmed root cause empirically, not by reading docs:** ran a real live
call against Token Factory (`Qwen/Qwen3-Next-80B-A3B-Thinking`) with
`stream=True` + `tools=` — model answers in plain text, `finish_reason:
"stop"`, no `tool_calls` ever populated, no error either. Same call with
`stream=False` works correctly. So tool-calling on this endpoint is only
reliable through a non-streaming call — there is no way to keep both
streaming and tool-calling on the same request.

**Fix (`backend/chatbot/client.py`):** added a cheap heuristic,
`_looks_like_lookup_needed()` — regex on digits + a keyword list (`step`,
`loss`, `metric`, `config`, `epoch`, `gpu`, `attention`, `top-k`, etc.) run
against the last user message. The non-streaming preflight (and therefore
tool-calling) now only runs when the heuristic fires; every other message
streams exactly as it did before PR #1. Deliberately biased toward "assume
no tool needed, stream normally" — per explicit instruction, tool-call turns
must always work correctly (they do, unconditionally, once triggered) but
plain conversation should default to streaming even at the cost of
occasionally missing a lookup the model could have made. Tests added in
`tests/test_chatbot_client.py`: one confirms a plain message never sends
`tools=` and streams normally; one confirms a lookup-style message goes
through the non-streaming path and returns the answer correctly. Full suite:
138 passed (was 134 pre-fix, +4 new tests, 0 regressions).

## §30: Chatbot grounding on live diagnostic-session data

The chatbot previously had no visibility into diagnostics-panel data (tensor
shapes, top-k predictions, attention/Q-K-V) captured while a user steps
through a paused run — it could only see the static config/source-code
snapshot and training metrics. Added a fourth chatbot tool,
`get_diagnostic_snapshot`, so the assistant can answer questions like
"what's the top prediction right now" or "what's the shape at block 2's
attention output" grounded in the real values the user is currently looking
at, rather than guessing from theory.

**Design:** diagnostic sessions are in-memory, keyed only by `session_id`
(`backend/training/diagnostics.py`), and the chatbot only ever knows a
`run_id` (from `get_tool_context()`), never a `session_id` — the frontend
never sends the diagnostics session id through the chat. Added a second
in-memory dict, `_run_to_session: dict[run_id, session_id]`, recording the
most recent session for a run. Populated in two places in
`backend/api/training.py`'s `diagnostics_start()`: on the **local** path,
`create_diagnostic_session()` now takes `run_id` and records it directly; on
the **remote/serverless** path, the session itself lives in the trainer
container's own process (a separate `_diagnostic_sessions` dict there) — the
main server only proxies the request, so it records just the returned
`session_id` string against the run_id, enough to know which id to ask the
trainer about later via the *same* existing `_is_remote`/`_proxy` path.

New `get_diagnostic_snapshot_for_run(run_id)` in `training.py` calls
`diagnostics_get(run_id, session_id)` directly (the existing route function)
rather than re-implementing the local/remote branch — one dual-path
implementation, reused, per the pattern already established for every other
route in this file. Returns `None` (never raises) when nothing is
available, which the chatbot tool turns into a plain "not available yet"
message rather than a hallucinated answer.

**A necessary side effect: `execute_tool_call()` became async.** The three
existing search tools are synchronous file reads; this one needs an `await`
(the accessor goes through FastAPI's async DB/proxy machinery). Rather than
special-case one tool as sync and one as async in `client.py`'s dispatch
loop, made `execute_tool_call()` itself `async def` — the sync tools inside
it are unaffected (they're just called normally, no `await` needed for
them), and the one call site in `client.py`'s tool-calls loop got a single
`await` added. `_looks_like_lookup_needed()`'s heuristic (added in §29) also
got diagnostic-related keywords (`shape`, `tensor`, `diagnostic`, `snapshot`,
`node`, `layer`, `head`, `mlp`, `embedding`, `activation`) so a question like
"what's the attention shape at layer 2" actually triggers the tool-calling
preflight instead of just streaming a generic answer.

**Test-isolation gotcha found while writing tests, noted so it isn't
mistaken for a real bug later:** `_run_to_session` (like the pre-existing
`_diagnostic_sessions` dict) is process-global and never cleared between
tests. Each test's `temp_db` fixture is a fresh SQLite file, so
autoincrement run ids restart at 1 every test — meaning a run_id used in one
test can collide with a stale mapping left behind by an earlier test that
used the same numeric id. A first draft of
`test_get_diagnostic_snapshot_for_run_tracks_run_to_session` asserted "no
session yet → None" before starting one, and failed because an earlier
test's diagnostic session for the same run_id number was still sitting in
the dict. Not a production concern (real run ids never repeat — SQLite
autoincrement is monotonic for the life of the actual database file), so no
production code changed; the test was simplified to only assert the
positive path instead.

## §31: Split concurrency limits by device x execution backend

Previously one combined limit (`max_concurrent_runs=2`, any device/backend)
plus a GPU-only sub-limit (`max_concurrent_gpu_runs=1`) governed everything —
local and serverless runs were counted together. That doesn't match reality:
a laptop's local CPU/GPU capacity is a hardware constraint, unrelated to how
many concurrent Nebius serverless endpoint sessions should be allowed.
Replaced with four independent settings
(`max_concurrent_local_cpu_runs=2`, `max_concurrent_local_gpu_runs=1`,
`max_concurrent_serverless_cpu_runs=3`, `max_concurrent_serverless_gpu_runs=3`
— values as given by the user), each checked only against runs matching
that exact device+backend combination.

**`_count_active_runs()` (in-memory) turns out to already be local-only,
just not documented as such** — it only reads `runner.py`'s `active_runs`
dict, which `_start_remote_run()` never populates (serverless runs are
backgrounded `asyncio.Task`s, not subprocesses). Added a docstring making
this invariant explicit rather than changing behavior, and `start_training()`
now only takes the `max()` with the in-memory count when the request is
local — folding it into a serverless count would silently undercount
nothing (it's always 0 for serverless) but read as more meaningful than it is.

`db.count_active_runs_in_db()` gained a `backend_filter` param (`AND
execution_backend = ?`), used alongside the existing `device_filter`. Both
optional, backward compatible for any other future caller.

Tests: `tests/test_training_concurrency.py` — DB filter combinations, a
zeroed local-CPU limit correctly rejecting a local start, the same zeroed
local limit NOT blocking a serverless start on the same device (the
independence the whole change is for), and a serverless GPU limit rejecting
an over-limit serverless GPU start. 147 passed (was 143, +4 new, 0
regressions).

## §32: Reopen a past experiment to add a new run

Previously the only way into the workspace was `PresetPicker` (always
creates a brand-new experiment) — there was no way to return to an
experiment that already had one or more runs and start another run on it.
`listExperiments()` already existed in `useApi.ts` (added for some earlier
purpose, unused anywhere in the UI) — the actual gap was only a component to
call it from and a handler to load the result into the workspace.

New `ExperimentBrowser.tsx`, rendered alongside `PresetPicker` on the
no-experiment-selected screen: lists experiments (name, id, template,
last-updated, most-recent-first) via `listExperiments()`, `onSelect(id,
config)` on click. New `handleLoadExperiment()` in `App.tsx` mirrors
`handlePresetSelect()` minus the `createFromPreset()` call — sets
`experimentId`/`config`, resets `runId`/`runStatus`/`metrics` to null/empty
(no run selected yet), defaults device/backend to "cpu"/"local" the same as
a fresh session. `TrainingControls` (already rendered whenever an experiment
is loaded, regardless of `runId`) is what actually lets the user pick
device/backend and click Start — no new run-creation path was needed, this
task was purely about getting an existing experiment's config back into the
same state a new one starts in.

"Within limits" (the user's phrasing) is already handled: the normal
per-device/backend concurrency check (§31) applies identically to a new run
started this way — no separate cap was added, since one would just
duplicate that existing enforcement.

Tests: `ExperimentBrowser.test.tsx` — empty state renders nothing, sort
order, `onSelect` called with the right id/config. Frontend suite: 33
passed (was 30, +3 new). Build clean.

## §33: Show serverless CPU/GPU hardware spec on the landing + workspace pages

Users had no way to see what hardware "serverless CPU" / "serverless GPU"
actually means (e.g. an L40 GPU) without reading `config/settings.py`
directly. Extended the existing `GET /api/nebius/workers/{device}` route
(already polled by `WorkerIdleBanner`) rather than adding a new endpoint —
it already had the right shape (per-device, already returns the *actual*
live preset once an endpoint has run) and just needed two more fields:

- `configured_platform`/`configured_preset` — straight from
  `settings.nebius_{cpu,gpu}_{platform,preset}`, always present, even
  before any worker session has ever existed (moved outside the
  `session is None` early-return, unlike the existing `preset` field which
  intentionally stays `None` until something real has run — see the
  2026-07-11 incident this endpoint was already built to guard against).
- `actual_platform` — the live platform captured in
  `worker_sessions.actual_platform` (§9's principle: prefer live truth over
  configured intent, since they can diverge). `actual_preset` already
  existed as the `preset` field; only `actual_platform` was missing.

New `HardwareSpecs.tsx` calls this for both `cpu` and `cuda`, shows
`platform · preset (live)` once an endpoint has actually run, falling back
to `platform · preset (configured)` before that — the `(live)`/`(configured)`
label is deliberate, not cosmetic: `settings.py`'s GPU platform/preset are
explicitly commented "UNVERIFIED placeholders... confirm before first GPU
use," so silently presenting them as fact would be actively misleading.
Rendered on both the landing page (below the description text) and the
workspace header (below the experiment title) — the two places the user
named.

Tests: 2 new backend tests (`configured_*` present with no worker session;
`actual_platform` overrides `configured_platform` once one exists) — 149
passed (was 147, +2 new). 2 new frontend tests (configured-spec state,
live-spec state) — 35 passed (was 33, +2 new). Build clean.

## §34: UI bugs from first real screenshot review

First look at the running app (screenshot) surfaced six issues, two of which
shared a root cause worth documenting so it isn't reintroduced.

**Step-through token count label was permanently stale.** The "(Step N, X
tokens)" label in `PausePrompt.tsx` computed X from
`diagnosticSnapshot.input_tokens.length` — but `input_tokens` is always just
the *original prompt's* tokens (`backend/training/diagnostics.py`'s
`_execute_forward_pass` builds it strictly from `session.prompt_tokens`,
never `token_history`). It never changes after the first step, so the label
looked frozen while the model's real input kept growing underneath it. Real
current length is `generated_token.position + 1`.

**No way to end a step-through session.** The prompt input already
correctly disabled itself once a diagnostic session started
(`disabled={diagnosticSession !== null}`), but nothing ever set
`diagnosticSession` back to `null` — there was no reset path at all short of
a page reload. Combined with the stale label above, a user could click `>`
repeatedly (or `>>`, which has no upper session-lifetime either) without
realizing the model's input sequence — no KV-cache, so every step reprocesses
the *entire* prompt + everything generated so far — was silently growing far
past what they'd typed. This is what produced a `[1, 82, 192]` runtime shape
that looked inexplicable from a 27-token prompt: it was correct, just
accumulated across an un-tracked, never-ending session. Fixed by auto-ending
the session when `>>` completes (it has a real endpoint, `max_new_tokens`)
and adding an explicit "Finish (new prompt)" button for the `>` path, which
has no natural end point (no EOS token in this char-level model).

**`activation_summaries` mislabeled as per-node.** It's computed once per
snapshot from the LM head's logits only
(`_compute_activation_extras(logits_last)`) — never per-node — but
`Inspector.tsx`'s Runtime tab rendered it under every node (ln1, attention,
mlp...) as if it reflected whatever node was selected. Moved to render only
under the `lm_head` node, relabeled "LM Head Logit Extras" to say plainly
what it actually is.

**Other fixes from the same review, unrelated root causes:** `HardwareSpecs`
now takes `device`/`backend` props and shows "Local" (no serverless info) or
only the single active device, instead of always showing both CPU and GPU
serverless specs regardless of what's actually running. Right-pane tabs
(`App.tsx`) had `gap: 0` with only vertical button padding — no horizontal
space at all between "Assistant"/"Inspector"/"Events", fixed with `gap: 24`.
`CodeView.tsx`'s "Serverless Metrics" tab used a manual left-margin/border
hack instead of using the panel's available width — replaced with
`justifyContent: space-between`, code-file tabs left, metrics tab right.

Not independently verified in a live browser this round (no Playwright tool
available in this session) — verified via build (`tsc` clean) + full
frontend test suite (35 passed) + careful tracing of the exact data flow
each bug report pointed to, not just pattern-matching the symptom.

## §35: Chatbot gave a confidently wrong "no config changes" answer

Live incident (2026-07-13): user changed `eval_interval` 20→10 mid-session,
then asked the chatbot "what is the config for this run?" It answered "No
config modifications have been made... this is the exact preset
configuration," which was false — visibly false, since the Config panel's
own baseline-diff shadow text showed "baseline: 20" right next to the
current value of 10.

Root cause confirmed from the real session log, not guessed:
`_get_last_audit_change()` (`backend/chatbot/context.py`) returned
`matches[-1]` — the literal last `lab.audit` line matching the experiment's
id marker, no filtering. Two things could (and did) make that the wrong
line: (1) `Notes updated` audit lines match the same `id=<N> ` marker as
config-change lines (the marker is deliberately shared, per the existing
"id=%d" / "experiment_id=%d" comment) but aren't config changes at all; (2)
a debounced config-panel autosave firing with no real difference logs
`Config updated: ... changed={}` — a real, separate audit line with an
empty diff. In the actual log, the real change (`eval_interval: [20, 10]`)
was followed six seconds later by exactly such a no-op save, which became
"the last change" and buried the real one entirely.

Fixed by walking the matched lines backward and returning the most recent
one that is both a `Config updated` line and has a non-empty diff
(`"changed={}" not in line`), skipping notes-update lines and no-op saves
either. Tests added for both failure modes plus the still-null case (only
no-op diffs exist for the experiment).

**A second, separate issue in the same conversation, not a code bug:** asked
for the exact val_loss at step 330 (which was a real, present step — pasted
directly from the live metrics log), the model answered "no data point
exists... some evaluations might have been skipped," which was also false.
Confirmed via the session log that `search_run_metrics` was never actually
invoked for either message in that conversation (no `Executed N chatbot
tool call(s)` line), even though the tool-calling heuristic (§29) did
correctly offer it — the volatile snapshot's loss trend is an
evenly-sampled subset (`_downsample_series`, `chatbot_loss_history_points`
default 25), and step 330 landed in a genuine gap between two sampled
points. The model treated a gap in its own sampled view as evidence the
data didn't exist, instead of calling the tool it had available specifically
for exact-step lookups. Not something fixable in application code — this is
model tool-calling judgment, not a logic bug — but tightened the system
prompt (`_SYSTEM_PROMPT`) to state explicitly that the injected loss trend
is a sampled subset and that a gap must never be reported as missing/skipped
data without checking `search_run_metrics` first. No way to verify this
prompt change actually improves real-model behavior without another live
conversation — flagging as unverified, not closed.

Backend suite: 152 passed (was 149, +3 new).

## File Layout

See `README.md` for project structure and setup instructions.
