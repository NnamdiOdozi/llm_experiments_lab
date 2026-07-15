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

## §36: >/>> now sample with temperature, matching Generate exactly

Previously `>` (step-through), `>>` (continue-generation), and `/generate`'s
token loop always picked the single highest-probability token
(`topk_ids[0]` / `torch.argmax`) — deliberately, on the assumption that
would keep the Top-k panel's rank-1 entry matching what actually got
selected. User pushback (2026-07-14): temperature never changes mid-run in
this app, so there's no real reason `>`/`>>` should behave differently from
Generate (`model.generate()`, which samples: `logits/temperature → softmax →
torch.multinomial`) — matching them isn't complexity, it's just using the
same recipe. Agreed; implemented plainly, no dynamic temperature-tracking
added (explicitly out of scope per the user — temperature is read once at
session start, same as it's read once for Generate).

**Change:** `DiagnosticSession` gained a `temperature: float = 0.8` field,
set once in `diagnostics_start()` from `config["inference"]["temperature"]`
— identical source to what `prompt_paused_model()` already uses for
Generate. `_execute_forward_pass()`'s token selection
(`backend/training/diagnostics.py`) and `/generate`'s inline loop
(`backend/api/training.py`) both replaced greedy pick with
`torch.multinomial(softmax(logits/temperature), 1)` — the exact recipe
`model.generate()` uses. The Top-k panel's own values are unaffected (still
raw, unscaled softmax(logits) — that's model confidence, not the sampling
knob) but the "selected" highlight in `Inspector.tsx` no longer assumes
rank #1 was picked (`entry.rank === 1`) — now matches by `token_id ===
generated_token.id`, with a fallback note when the sampled token falls
outside the top 5 entirely (which sampling makes possible and is not an
error).

Removed the "greedy vs sampling, see caption" explanatory text added
earlier the same day in `PausePrompt.tsx` — no longer true, so no longer
needed.

**Trainer-image note:** this touches `backend/training/diagnostics.py` and
`backend/api/training.py` — both run inside the trainer container for real.
Requires a rebuild + repush to take effect on serverless runs, unlike the
frontend-only fixes earlier the same day.

Tests: `test_diagnostics_start_reads_temperature_from_config`,
`test_diagnostics_start_defaults_temperature_when_config_omits_it`,
`test_diagnostic_step_samples_instead_of_greedy` (spies on
`torch.multinomial`, asserts it's actually called), and
`test_generate_samples_instead_of_greedy` (same, for `/generate`'s separate
loop — 3 max_new_tokens → 3 multinomial calls). Backend suite: 156 passed
(was 152, +4 new). Frontend: 35 passed, build clean.

## §37: Inspector redesign — Block/Head moved into Inspector, bar chart, clearer LM head labels

Several rounds of live UI feedback (2026-07-14), landed together since they
all touch the same panels:

**Block/Head/Q-K-V-detail moved from Prompt Model into the Inspector pane.**
Previously "Layer"/"Head" number inputs lived in `PausePrompt.tsx`,
disconnected from the Architecture diagram where you actually pick which
block you care about — user had to keep two mental models of "block" in
sync (the diagram's "Block 4 of 4" labeling vs. a separate 0-indexed
"Layer" number field), and initial live testing suggested attention data
"wasn't coming through" when in fact it was a workflow gap (verified by
directly driving the real API in-process — `POST diagnostics/start` →
`POST .../step` with `attention_layer=0, attention_head=0` returned a
correct, full 14x14 weight matrix — the backend was never the problem).

Fix: **Block is no longer a separate input at all.** `App.tsx` derives it
from whichever attention node is currently selected in the diagram
(`selectedNodeId.match(/^block\.(\d+)\.attention$/)`) — selecting the node
in the diagram *is* selecting the block. Only **Head** remains a manual
input (the diagram can't show which head), now rendered inside
`Inspector.tsx`'s Runtime tab, only when an attention node is selected —
contextual to where the data actually appears. `PausePrompt.tsx` no longer
owns any attention-selection state; `attentionBlock`/`attentionHead`/
`showQKVDetail` are lifted to `App.tsx` and passed down to both components.
This also fixes the actual root confusion in the original bug report: the
"one little grey square" the user saw was the QKV-detail checkbox rendering
unstyled (native browser widget, no theming) — likely still there in older
screenshots, but with Head/QKV now living in the Inspector next to the data
they configure, disconnected placement is no longer possible by
construction, not just prevented by better labeling.

**Top-k Tokens got a bar chart** (probability-proportional width, green for
whichever token was actually selected by sampling) — was plain text, per
explicit ask ("much more visual").

**Removed "Top Absolute Components" entirely**, rather than relabeling it.
It ranked the same `logits_last` vector Top-k already shows, but by
`|logit|` (magnitude, sign ignored) instead of probability — a genuinely
different criterion that pulls in strongly *negative* (very unlikely)
logits alongside likely ones, producing a different order than Top-k for no
clear pedagogical reason. Two overlapping "top 5" lists from the same
vector with different orderings was the actual confusion, not just a
labeling problem — relabeling would have left that intact.

**"Value Slice (first 8)" renamed and clarified.** It's `logits_last[:8]`
— the LM head's own logits, first 8 vocab-index positions, in vocab-index
order (not ranked). Easy to misread as the hidden dim (192, the number
shown everywhere else in this panel) when it's actually vocab_size (65).
Now reads "LM Head Logits (first 8 of {vocab_size}, raw — vocab index
order, not ranked)", with the count read live from
`snapshot.lm_head.logits_shape` rather than hardcoded.

**"Not captured" messages made actionable**, replacing a single generic
string reused for several different actual states: no diagnostic step run
yet ("enter a prompt and click > below"), attention not requested for this
step ("pick a head above, then click > again").

All frontend-only — `PausePrompt.tsx`, `Inspector.tsx`, `App.tsx`. No
trainer-image impact. Build clean, 35/35 frontend tests pass (no test
covered the removed/relabeled Inspector internals directly, so none needed
updating).

## §38: >> ignored both attention params and config's max_new_tokens

Two related, now-fixed gaps in `generateDiagnosticStream()`
(`frontend/src/hooks/useApi.ts`) / its call site in `PausePrompt.tsx`'s
`handleContinueGeneration`:

**Attention/Q-K-V never sent.** The request body only ever included
`max_new_tokens` — `attention_layer`, `attention_head`, `qkv_detail` were
never sent at all, even though `>` correctly sent them and the backend
route (`DiagnosticsGenerateRequest`) always accepted them. Whatever was
picked in Inspector (§37) was silently dropped for `>>`. User asked
directly whether Inspector's selection genuinely drove both `>` and `>>` —
it didn't, for `>>`. Fixed: `generateDiagnosticStream()` gained an optional
4th param (`DiagnosticStepRequest`, reusing the existing type), spread into
the request body; `PausePrompt.tsx` now passes
`attentionBlock`/`attentionHead`/`showQKVDetail` through. Verified two ways,
not just asserted: a unit test on the actual `fetch` call body, and a live
in-process API call (real `httpx` client, real model, no mocking) through
the real `/generate` route with `attention_layer=2, attention_head=3,
qkv_detail=true` → confirmed real attention data + qkv_detail in the
response.

**`max_new_tokens` hardcoded to 50, ignoring `config.inference.max_new_tokens`.**
User caught this directly (dashboard showed 100). Confirmed via `git log
-p` that the `50` predates this session — not something introduced while
fixing the attention-params gap above, just sitting on the same line I was
already editing and worth fixing while there. Unlike temperature (§36,
which is session-level — applies to every token in both `>` and `>>`, so
it's read once into `DiagnosticSession` at `diagnostics_start()`),
`max_new_tokens` only matters for `>>`'s burst size and is already a
per-request parameter on the `/generate` route — no backend/session change
needed. Fixed frontend-only: `App.tsx` passes
`config?.inference?.max_new_tokens` (falling back to 50 if the config
lacks it) down to `PausePrompt` as a `maxNewTokens` prop, replacing the
hardcoded literal — same value the Generate button's `/prompt` route
already reads server-side via `inference_cfg.get("max_new_tokens", ...)`
in `runner.py`.

Both fixes are frontend-only (`useApi.ts`, `App.tsx`, `PausePrompt.tsx`) —
no trainer-image impact. Tests: `useApi.test.ts` (new, 2 tests) — asserts
the real request body sent to `fetch`, both with and without attention
params. Frontend suite: 37 passed (was 35, +2 new). Build clean.

## §39: Inspector overhaul — per-position capture, heatmap color scale, 1-indexing, dropdown

Large batch (2026-07-14), scoped via explicit back-and-forth before coding
per the user's request — plan converged over several exchanges (table vs.
stepper per section, position-window cap, backend cost) before any code
was written.

**Backend (`diagnostics.py`, trainer-image change):**
- `DIAGNOSTIC_POSITION_WINDOW = 12` — caps both new per-position datasets to
  the most recent 12 positions, regardless of how long the session's
  sequence has grown (directly guards against the earlier "82 tokens
  accumulated silently" scenario, §36/37).
- `lm_head.top_k_by_position` (new, additive — existing `top_k` untouched):
  per-position top-5, free in compute terms since `logits` already covers
  every position in the one forward pass already being done — previously
  only the last position's top-k was kept, the rest silently discarded.
- `attention.qkv_detail` (restructured, breaking change to that field only
  — low blast radius, no other consumers): was one vector per Q/K/V for the
  *last* token only; now `positions`/`tokens`/`q`/`k`/`v` arrays, one entry
  per position in the window.
- **Documented, not "fixed"**: `top_k_by_position`'s window ends one
  position earlier than `qkv_detail`'s, because `top_k_by_position` is
  computed before this step's new token is appended to `token_history`,
  and `qkv_detail` (via `_compute_attention_weights`) after. That's
  correct, not misaligned — the just-generated token has real Q/K/V (it's
  part of the sequence now) but no "what comes next" prediction yet (that
  needs another forward pass, which is exactly what the next step
  produces). Each dataset carries its own explicit `position` labels for
  this reason, not a shared implicit index.

**Frontend — two different UI patterns for two different data shapes,**
decided via direct back-and-forth (a Colab Variable Inspector screenshot
prompted the pivot from an initial single stepper plan):
- **Q/K/V → `QKVTable`**: all positions in the window as table rows (Position/Token/Q/K/V), each vector cell truncated (first 4 values) with the full vector on hover (native `title` tooltip — same lightweight mechanism the heatmap cells already used, no new UI dependency). Whole window visible at once.
- **LM Head → `LmHeadStepper`**: explicit ◀ position N of T ▶ control, full ranked top-5 + bars for the selected position — user explicitly rejected the table pattern here ("too much to see at once" — a 5-entry ranked list with bars doesn't compress into one truncated cell usefully). Defaults to the most recent position, numerically identical to the old fixed single-position view.

**Attention heatmap color scale, real fix not just relabeling:**
normalization moved from global min/max to **per-row** (each row is its own
probability distribution — the meaningful comparison), added a minimum
opacity floor so no real value is literally invisible, and causally-masked
cells (`j > i`, structurally always exactly 0) get a distinct hatched style
instead of being mixed into the same color scale as genuine low-attention
values. Root cause of "one white cell and nothing else," reported three
times before finally being traced instead of explained around: `(0,0)` is
mathematically guaranteed to be exactly `1.0` for any layer/head (first
token can only attend to itself under causal masking) — confirmed live —
so it dominated the old global max and pushed almost every other cell's
opacity near zero. Also added an explicit "Q\K" label to the table's corner
cell, which was blank/ambiguous before and looked like a data cell.

**Stale-data indicator (real bug report, "Layer/Head not updating"):**
changing Head (or clicking a different block's node) only updates local
selection state — it's never retroactively applied to an already-captured
snapshot, only the *next* `>`/`>>` click. Added a visible banner comparing
currently-selected Block/Head against what the displayed snapshot actually
captured (`snapshot.attention.layer`/`.head`), telling the user to
re-click `>` rather than silently showing stale data with no explanation.

**1-indexed display, 0-indexed everywhere internal:** Head is now a
`<select>` (bounded to `1..config.model.n_head`, sourced live not
hardcoded) instead of a number spinner with barely-visible native arrows —
fixes both the indexing complaint and the "poor quality widget" complaint
in one change, since a dropdown for a small bounded set is the correct
widget choice independent of the indexing issue. Block is still never a
separate input (derived from `selectedNodeId`, per §37) — just displayed
as `+1` everywhere a human reads it (Inspector heatmap header, PausePrompt
status line). Internal state, request payloads, and array indexing stay
0-indexed throughout — this is a display-layer conversion only.

Frontend suite: 37 passed (unchanged — no existing test asserted the old
Inspector internals being replaced). Backend suite: 157 passed (unchanged
from the per-position-capture work in §38... this section — no backend
logic changed further in the frontend-only portion of this batch). Build
clean.

## §40: Auto-refresh attention on Head/Block change + LM head stepper tracks current step

Two follow-ups from live use of §39's Inspector overhaul.

**LM head stepper now auto-advances.** `LmHeadStepper`'s position `index`
was local `useState`, initialized once — after the user manually stepped
the ◀/▶ control, it never moved again on its own, so clicking `>` in Prompt
Model (a genuinely new step, new snapshot) left the stepper pointing at a
stale index instead of following the newest position. Added a `useEffect`
keyed on `snapshot.generation_step` that resets to the latest position
whenever a new step actually happens — manual navigation between steps
still works, it just re-syncs to "latest" each time the model actually
advances.

**Real fix, not cosmetic: changing Head in Inspector now actually refreshes
the heatmap**, instead of requiring a `>` click. Root cause was structural,
not a missing event handler: attention data only ever got (re)computed as
part of `run_diagnostic_step()`, which always samples and appends a new
token — there was no way to ask "recompute attention for a different
head, same position" without also advancing the model. The fix reuses
`run_diagnostic_step_internal` (already existed, built for `/generate`'s
final-frame capture, `skip_token_generation=True` → no token sampled,
`session.generation_step`/`token_history` untouched) via a new route,
`POST /{run_id}/diagnostics/{session_id}/peek` — same local/remote
`_is_remote`/`_proxy` dual-path structure every other diagnostics route
uses. Frontend: `App.tsx` now tracks the active session id (surfaced by
`PausePrompt` via a new `onSessionIdChange` callback — previously private
to that component) and a `useEffect` calls `peekDiagnostic()` automatically
whenever `attentionBlock`/`attentionHead`/`showQKVDetail` change, updating
`diagnosticSnapshot` without touching `diagnosticStep` (a peek is not a
step). The stale-data banner from §39 stays as a fallback — if a peek
request fails (network error, or a serverless run whose trainer container
predates this change and 404s), the banner still correctly flags the
mismatch instead of silently showing wrong data.

**Trainer-image note:** `/peek` is a new route in `backend/api/training.py`
— needs a rebuild+repush for serverless runs. Same category as §36/§38.

Tests: `test_peek_recomputes_attention_for_a_new_head_without_advancing`
(confirms `generation_step` and token count both stay unchanged across two
different peeks), `test_peek_before_any_step_still_works` (works on a
fresh session with no prior `>` click). Backend suite: 159 passed (was
157, +2). Frontend: `useApi.test.ts` gained a `peekDiagnostic` request-body
test — 38 passed (was 37, +1). Build clean.

## §41: Generic per-node vectors + LM head stepper "X of Y" fix + a real MoE bug found along the way

**Generic node vectors** — the "generalize hover-inspection to every node"
feature explicitly deferred earlier is now built, scoped via direct
confirmation before coding (capture-every-node-every-step over
capture-only-selected, accepting the larger response size; same
Colab-style table as Q/K/V). Every hook (embedding, ln1, attention-out,
ln2, mlp/moe, final_norm) previously discarded its raw output right after
computing shape+summary stats — `NodeCapture` gained a `position_vectors`
field (`{positions, vectors}`, capped to `DIAGNOSTIC_POSITION_WINDOW`, same
shape convention as `qkv_detail`), populated in the shared hook closure in
both `transformer/model.py` and `moe/model.py` for any 3D `[B, T, *]`
output. Frontend: `NodeVectorTable` — one column instead of QKVTable's
three, no token-text column (position numbers alone correlate against the
heatmap/top-k tables already shown; re-decoding tokens at ~18 nodes every
step for a column shown elsewhere already wasn't worth the payload).

**A real, separate bug found while implementing this** — not something I
went looking for. The MoE template's diagnostic hooks
(`moe/model.py::register_diagnostic_hooks`) only ever branched on
`isinstance(output, tuple)`, true only for the `.moe` node itself (which
genuinely returns `(x, drop_rate)`). Every other node in the MoE template —
embedding, ln1, attention, ln2, final_norm, lm_head, i.e. 6 of 7 node types
— returns a plain tensor, fell into the `else` branch, and silently got
`output_shape=[]`, `summary={}`. This has presumably been true since the
MoE template's diagnostics were first built; nothing caught it because no
test exercised MoE diagnostic capture at all. Fixed by adding the missing
`elif isinstance(output, torch.Tensor)` branch, mirroring the transformer
template's (correct) handling.

**LM head stepper "X of Y" was showing two contradictory numbers.** Real
bug report with a screenshot: a 15-token sequence's last position correctly
labeled "Position 14," but right next to it "12 of 12" — because
`entries.length` is the *windowed* array size (capped to
`DIAGNOSTIC_POSITION_WINDOW`=12), not the true sequence length, and the
label never said so. Fixed: now reads "12 of 12 shown (last 12 of 15
total)" when windowed, computing the true total from
`generated_token.position + 1` (same value used for the §36 token-count
fix) rather than a second, uncontextualized count.

Tests: `test_generic_node_captures_position_vectors` (all non-lm_head
nodes get real per-position vectors, correct `n_embd` width),
`test_moe_template_captures_all_nodes_not_just_moe_block` (regression test
for the MoE bug — asserts every node has non-empty shape/summary, not just
`.moe`). Backend suite: 161 passed (was 159, +2). Frontend: 38 passed
(unchanged — no test asserted the fixed stepper label). Build clean.

**Trainer-image note:** both `transformer/model.py` and `moe/model.py`
changed — needs a rebuild+repush, same category as every backend change
this session.

## §42: Greedy/sampling toggle, input vectors everywhere, 1-indexed positions, no-blank Head dropdown, double-click data tabs

Six-item batch from one round of direct feedback (2026-07-15), driven by
live use of the Inspector, not upfront spec.

**1. Greedy vs sampling decoding mode.** New `inference.decoding_mode`
config field (`"sample"` | `"greedy"`, default `"sample"` — unchanged
default behavior). Read once at diagnostic-session creation
(`DiagnosticSession.decoding_mode`) and by `/generate`'s inline
token-selection loop and `prompt_paused_model()` — same source Generate
already used for `max_new_tokens`/`temperature`, so `>`, `>>`, and Generate
all honor it identically. `model.generate(..., greedy=greedy)` added to all
three templates (transformer, moe, rnn): `torch.argmax` under greedy,
`torch.multinomial(softmax(logits/T))` under sample — same branch already
used in `_execute_forward_pass`.

Temperature is mathematically inert under greedy: `argmax(logits/T) ==
argmax(logits)` for any positive T, since dividing preserves rank order.
`ConfigPanel.tsx` greys the `temperature` field and shows "no effect under
greedy" when `decoding_mode === "greedy"`, rather than leaving a live
control that silently does nothing.

**2. Input vectors alongside output vectors, for every node.** Previously
`NodeVectorTable` only showed a node's output. Real gap: no way to see
what a LayerNorm actually *changed*, just what it produced. Every hook
(`register_diagnostic_hooks` in both transformer and moe templates) now
also captures `input_position_vectors` via the same windowed-capture
helper used for outputs. `NodeVectorTable` renders an Input column
whenever `inputPv` is present; omitted for nodes whose input isn't a
per-position float vector (e.g. embedding, whose input is token ids).

**3. Consistent vector truncation: first 6, "…", last 6, everywhere.**
Previously ad hoc — one place showed only the first 4. `truncatedVector()`
is now the single shared helper behind both `QKVTable` and
`NodeVectorTable`; full-precision value still available via the `title`
hover tooltip (unchanged pattern) and now also via the new data tab (item
6 below).

**4. All displayed position numbers are 1-indexed.** Heatmap row/column
headers, `QKVTable`/`NodeVectorTable` Position columns, and
`LmHeadStepper`'s "Position N" label now all render `position + 1`.
Internal state, array indices, node ids, and every backend request/response
payload stay 0-indexed — this is a display-only conversion at the point of
JSX rendering, same convention already established for Block/Head numbers
in §37. Chosen specifically because the user flagged inconsistent
indexing ("some places start from 0... it's just odd") as its own bug
class, not something to fix piecemeal per screen.

**5. Head dropdown no longer defaults to a blank option.** Real reported
friction: "I don't think we should be starting on a blank entry where
there's no image. It just frustrates the user." `Runtime()` now has a
`useEffect` that defaults `attentionHead` to `0` the moment an attention
node is selected and nothing's picked yet (placed before any early
`return`, per the Rules of Hooks). The `<select>` no longer renders an
empty first `<option>`.

**6. Double-click a vector cell → opens a new, closeable Data tab.**
Direct reference to the Colab/VS Code variable-inspector pattern: double-
click to see the whole value, static and copyable. Implementation:
- `App.tsx`: `RightPaneTab` widened from a fixed union to `string` so
  dynamically-generated tab ids type-check. New `DataTab { id, title,
  content }` state array (`dataTabs`), `openDataTab(title, content)`
  (generates a unique id, appends, switches to it) and
  `closeDataTab(id)` (removes it, falls back to `"assistant"` if it was
  active). Tab bar renders one button + a "×" close button per open data
  tab, appended after the static Assistant/Inspector tabs. Tab content
  renders the full value in a read-only `<textarea>` (not `<pre>` —
  needs to be a proper scrollable, selectable, copyable text control).
- Each double-click opens a **separate new tab**, not a shared/reused one
  — confirmed twice by the user, explicitly rejecting the "replace a
  single tab" alternative, specifically so multiple values can be compared
  side by side and closed individually rather than being clobbered by the
  next double-click.
- `onOpenDataTab` threaded from `App.tsx` → `Inspector` → `Runtime` →
  `QKVTable`/`NodeVectorTable`, called from each `<td>`'s `onDoubleClick`.
  Tab titles carry enough context to identify the value without opening
  it: `QKVTable` uses `Block {n} Head {n} — Q/K/V — pos {n}`;
  `NodeVectorTable` uses `{nodeId} — Input/Output — pos {n}` (nodeId is
  already a stable, descriptive string like `block.2.mlp` or
  `final_norm`).

Frontend suite: 38 passed, build clean. Backend suite: 166 passed, no
regressions (items 1-2 are covered by the existing decoding-mode/input-
vector tests added alongside the code; items 3-6 are frontend-only, no
backend tests apply).

**Trainer-image note:** items 1 and 2 touch `backend/training/*` (all
three template `model.py` files, `diagnostics.py`) and
`backend/api/training.py` — needs a rebuild+repush before it's live on
serverless runs. Items 3-6 are frontend-only and need no rebuild.

## §43: Pause-after-completed 502, stale prompt survives resume, embedding table, and Inspector layout requests

Two real bugs found live tonight (2026-07-13), traced with server logs and
`git blame` rather than guessed at, plus a batch of confirmed UX requests.

**Bug 1 — pausing an already-completed run returned a bare 502.** Real
user report: clicked pause right as a run hit 1000/1000 steps (frontend's
step counter hadn't caught up with the last poll yet). `pause_training()`
in `backend/api/training.py` proxied straight to the remote endpoint
regardless of local status. The remote's own `pause_run()` correctly
returned 400 "not running," but `_proxy()`'s `raise_for_status()` turned
that 400 into an `httpx.HTTPStatusError`, which the wrapper collapsed into
a generic, unhelpful `HTTPException(502, ...)` — discarding the real
reason entirely. Fixed: check `db_run["status"] in TERMINAL_STATUSES`
*before* ever touching the network, returning a clear
`400 "Cannot pause — run is already completed"` locally. Test:
`test_pause_training_on_completed_remote_run_returns_clear_400` — asserts
zero proxy calls happen at all.

**Bug 2 — prompting during a pause, then resuming, then pausing again left
the old partially-stepped-through prompt/session in place**, and stepping
it errored. Root cause: `PausePrompt.tsx` never unmounts when `canPrompt`
flips false (App.tsx keeps it mounted for the run's whole lifetime), so
its `diagnosticSession`/`prompt`/`generatedTokens` state survived the
round trip untouched. Server-side, resuming loads a fresh model checkpoint
into a new process, so the old in-memory diagnostic session's `model`
reference pointed at nothing valid anymore — stepping it threw. Fixed: a
`useEffect` keyed on `canPrompt` clears all of that state (session,
snapshot, prompt text, output, generated tokens) the instant the run
leaves paused/completed. Resuming or retraining now always hands you a
clean slate for the next prompt.

Both traced and fixed the same night they were reported — verified via
`git blame` that bug 2's origin (`1e6cd6a`, 20:15 that day, the §40
auto-refresh-attention work) was mine, from earlier in the session, not
the batch immediately before it, per the user's direct question ("it will
have been one of the last changes you've put through").

**UX batch**, all direct user requests, 2026-07-13:

1. **"Metrics" not "Serverless Metrics."** `CodeView.tsx`'s tab worked
   identically for local and remote runs already (`WorkerPanel.tsx` has no
   nebius-specific branching — cpu/ram/gpu fields just come from whatever
   the run actually reports) — the label was just wrong, not the content.
   Simple rename, no split into two tabs needed.
2. **Data tab returns to the tab that opened it, not always "assistant."**
   `DataTab` gained a `returnTo` field, captured from `rightPaneTab` at
   the moment `openDataTab()` is called; `closeDataTab()` restores it
   instead of hardcoding `"assistant"`.
3. **Data tab is now an actual index/value dataframe**, not a bracketed
   string blob. `DataTab.content` changed from a preformatted `string` to
   a raw `number[]`; every `onOpenDataTab` call site in `Inspector.tsx`
   (`QKVTable`, `NodeVectorTable`, the new `EmbeddingTable`) now passes the
   raw vector instead of `truncatedVector(...).full`. Rendered as an
   Index | Value table, one row per element, scrollable.
4. **Heatmap shows the actual token character, not the position number.**
   `AttentionHeatmap`'s row and column headers now render `token`/
   `tokenLabels[i]` directly; position is still available via the `title`
   hover tooltip. (TS note: had to bind `att.token_labels` to a local
   `const tokenLabels` after the `!att.token_labels` guard — narrowing
   through `att.token_labels` directly doesn't survive into the nested
   `.map()` closures.)
5. **Input above Output; Q above K above V — vertical, not side-by-side
   columns.** New shared `VectorPreviewTable` component (one vector-kind,
   Position(+Token)/Value) used by both `QKVTable` (three stacked
   instances: Q, K, V) and `NodeVectorTable` (two stacked instances:
   Input, Output). Replaces the old wide multi-column single table.
6. **LM Head stepper label simplified.** Was `Position 20 ("n") — 12 of 12
   shown (last 12 of 21 total)`; now just `Position 20 ("n")`. The window
   explanation was judged unnecessary noise once the ◀/▶ controls are
   already visible — real feedback, 2026-07-13.
7. **Embedding table/matrix in the Inspector.** New backend route
   `GET /{run_id}/architecture/embedding-table` (local+remote proxy, same
   pattern as `/architecture`) — loads the *actual trained checkpoint*
   (unlike `/architecture`, which only builds a fresh untrained model to
   count params) and returns `model.token_emb.weight` in full
   (vocab_size × n_embd — small enough for these char-level toy models
   that no windowing was needed) alongside each row's decoded character.
   transformer/moe only — RNN's `CharRNN` has no `token_emb` (one-hot
   input instead), returns a clear 400. New `EmbeddingTable` component in
   `Inspector.tsx`, fetched once per `runId` and shown under the
   `embedding` node's Runtime tab, reusing the same double-click-to-open
   pattern as everything else. `Inspector`/`Runtime` both needed a new
   `runId` prop threaded down from `App.tsx` to support this.

Tests: `test_get_embedding_table_returns_real_trained_weights` (writes a
known, distinctive value into the checkpoint's embedding matrix and
confirms the route returns *that* value, not a freshly-initialized
model's), `test_get_embedding_table_404_without_checkpoint`,
`test_get_embedding_table_rejects_rnn`,
`test_pause_training_on_completed_remote_run_returns_clear_400`. Backend
suite: 170 passed (was 166, +4). Frontend: 38 passed, build clean.

**Trainer-image note:** the new embedding-table route lives in
`backend/api/training.py` — needs the same rebuild+repush as every other
backend change this session before it works on a serverless run whose
trainer image predates tonight's work (see §42's note — this is the exact
failure mode that was live-diagnosed via the CPU run screenshot earlier
tonight). Everything else in this batch is frontend-only.

## §44: Embedding node shows one-hot input, windowed+steppable attention heatmap, LM head label trimmed

Direct user requests, 2026-07-13, continuing straight on from §43.

**1. Embedding node's Runtime tab now shows one-hot INPUT vectors, not
output vectors.** Previously showed `NodeVectorTable`'s float output
vectors (same as every other node) — user wanted the *input* instead:
"position, then the character, and then the one-hot vector... just leave
it at that for that tab." New `EmbeddingOneHotTable` component replaces
`NodeVectorTable` entirely for the `embedding` node (not shown alongside
it). Synthesized client-side — a one-hot vector is fully determined by the
token id alone (`vocabSize`-wide array, 1 at that index, 0 elsewhere), no
new per-node backend capture needed. Requires knowing the real token id at
each position, which the snapshot didn't previously expose beyond the
prompt-only `input_tokens` and the single most recent `generated_token` —
added a new top-level `position_tokens: [{position, id, token}, ...]`
field to `DiagnosticSnapshot`, computed once in `_execute_forward_pass`
from the same pre-append window already used for `top_k_by_position`
(shared across every node in one forward pass, so it's one list, not
duplicated per node). The (separate, still-present) full embedding
*weight matrix* view added in §43 is unaffected — this is about the
per-position input to the layer for the current prompt, not the trained
table itself.

**2. Attention heatmap is now windowed and steppable.** Real bug found
while implementing the fix: the heatmap's `weights`/`token_labels` were
*never* capped at all — unlike `qkv_detail`, which was already windowed to
`DIAGNOSTIC_POSITION_WINDOW`, the heatmap rendered the full, ever-growing
T×T attention matrix. Exactly matches the user's report ("gets very busy
in the inspector very quickly"). `_compute_attention_weights` now takes a
`window_offset` param (0 = most recent window, positive N = shift the
window's end back N positions) and slices `weights` to a square
`DIAGNOSTIC_POSITION_WINDOW`-sized block on *both* axes — not just rows —
so a long session shows a fixed-size, readable grid instead of an
unbounded one. `qkv_detail` now shares the exact same window (previously
computed its own independent trailing slice; now both come from one
`start`/`end` pair). Threaded end-to-end: `DiagnosticsStepRequest` gained
`attention_window_offset: int = 0`; `run_diagnostic_step`/
`run_diagnostic_step_internal`/`_execute_forward_pass` all pass it
through; the `/step` and `/peek` routes forward it (both locally and via
the remote proxy body). `AttentionHeatmap` gained a ◀ Earlier / Later ▶
stepper (only rendered when the sequence is actually longer than one
window) showing "Positions N–M of T", driven by new `attentionWindowOffset`
state in `App.tsx` — reset to 0 whenever `runId` or `selectedNodeId`
changes (a stale offset from a previously-viewed block/head silently
carrying over would show the wrong slice). `>` step calls now also send
the current offset (so stepping while viewing an earlier window captures
that window, not always the tail); `>>` continue-generation deliberately
does **not** — it already produces only a single final snapshot, and
window-shifting is a review action that makes more sense against a
snapshot already at rest via `/peek`, not mid-stream.

**3. LM Head stepper label trimmed.** Was `Position 20 ("n") — 12 of 12
shown (last 12 of 21 total)`; the trailing "X of Y shown" clause read as
noise once the ◀/▶ controls are already visible (real feedback, same
session as items 1-2). Now just `Position 20 ("n")`.

Tests: `test_attention_heatmap_windowed_not_full_matrix` (confirms the
heatmap is capped to a `DIAGNOSTIC_POSITION_WINDOW`² block, not the full
matrix — regression test for the just-found unbounded-heatmap bug),
`test_attention_window_offset_shifts_window_earlier` (confirms `/peek`
with an offset shifts `window_start` by exactly that much),
`test_snapshot_includes_position_tokens_for_embedding_one_hot`. Backend
suite: 173 passed (was 170, +3). Frontend: 38 passed, build clean.

**Trainer-image note:** items 1-2 touch `backend/training/diagnostics.py`
and `backend/api/training.py` — same rebuild+repush requirement as every
other backend change this session (see §42, §43). Item 3 is frontend-only.

## §45: Position embedding table, embedding output vectors restored, vocab_size read-only

Direct follow-up, 2026-07-13, immediately after §44 — course-correction on
the embedding node's Runtime tab, plus one unrelated config fix.

**Position embedding table added; output vectors restored alongside the
one-hot input.** §44 replaced the embedding node's output-vector table
with the one-hot input table — user then clarified they wanted *both*,
not one replacing the other, and also wanted the position embedding
matrix (not just the token embedding matrix already added in §43).
`GET /{run_id}/architecture/embedding-table` now also returns
`block_size`/`position_embedding` — extracted from `model.pos_emb.weight`
via `hasattr(model, "pos_emb")` (the actual loaded model, not the config
string, so it can't drift out of sync with what the checkpoint really
contains). Only present under `pos_encoding="learned"`
(`nn.Embedding(block_size, n_embd)`, a real trained parameter); `null`
under `"rope"`, since rotary position encoding is computed on the fly and
has no table — the frontend shows a clear "Not applicable — this model
uses RoPE" message in that case rather than an empty table. The embedding
node's Runtime tab now shows, top to bottom: one-hot input vectors →
output vectors (real, captured) → token embedding table → position
embedding table (or the RoPE note).

**`vocab_size` is now read-only in the Config panel.** It's a fact about
the dataset (character-level, fixed alphabet), not a real training
choice — editing it doesn't do anything useful and just risks a confusing
mismatch. `ConfigPanel.tsx` gained a `READ_ONLY_FIELDS` set (same
disabled/title/footnote pattern already used for temperature-under-greedy)
rather than hiding the field — still shown for information, just not
editable.

Tests: `test_get_embedding_table_includes_position_table_when_learned`,
`test_get_embedding_table_omits_position_table_when_rope`. Backend suite:
175 passed (was 173, +2). Frontend: 38 passed, build clean.

**Trainer-image note:** the embedding-table route change touches
`backend/api/training.py` — same rebuild+repush requirement as every other
backend change this session. The `vocab_size` read-only change and the
Runtime tab restructuring are frontend-only.

## §46: Full front-to-back trace of Generate/>/>>, config staleness bug, Generate removed entirely

2026-07-14. User reported three tangled symptoms — >> stopping around 50
tokens instead of 100, Inspector showing stale/mismatched positions, and a
confusing double-line "Output so far" box — and asked for a slow, careful
trace rather than another guess. Did exactly that: direct API calls
against the live backend (bypassing the browser entirely, using
`urllib.request` against `localhost:8000`) to isolate backend behavior
from frontend/browser behavior, plus reading `diagnostic_sessions` DB rows
and the session log as ground truth for what actually happened in the
user's own browser.

**Root cause of the >> token-count bug: `config` never re-syncs from the
server after page load.** `App.tsx` seeds `config` state once from
`sessionStorage` (`saved.current?.config`), which survives a hard refresh
— only clearing when the tab closes. An existing `useEffect` already
fetched the live experiment on load, but only to compute the ConfigPanel
baseline-diff display; it threw away `exp.config` instead of using it to
correct a stale cached copy. Traced the *exact* request the user's browser
sent (`diagnostic_sessions` row, timestamp cross-checked against the
UTC/BST offset to confirm it was really their click): `max_new_tokens: 50`
— sent even after a hard refresh and a backend restart, while the
Config panel visibly showed 100 and the server's saved config genuinely
was 100. ConfigPanel computes its own display value locally
(`normalizedConfig` merges in `INFERENCE_DEFAULTS` for display purposes)
— that's what showed 100 — but the actual `config` state feeding `>>`'s
request body was a frozen, older copy. Fixed: the same experiment-load
effect now calls `setConfig(exp.config)`, so every experiment load
re-syncs to the server's real, current value rather than trusting
whatever was cached.

Also disproved two of my own earlier theories in the process, worth
recording so they aren't re-investigated later: (1) "stale JS bundle from
before a fix" — ruled out, since the bad value persisted through a
confirmed-fresh hard refresh; (2) "browser abandoning the SSE stream
early" — ruled out by direct API trace: a 100-token `>>` call, run
directly against the backend, completed in ~24s with exactly 100 token
events and zero drops. The suspicious "200 (200ms)" request-log lines for
`/generate` calls turned out to be a red herring — `backend/main.py`'s
logging middleware measures how fast the endpoint *returns a
StreamingResponse object* (near-instant, since it just wraps a generator),
not how long the stream takes to actually finish sending.

**Generate button removed entirely — direct user decision, confirmed via
AskUserQuestion.** Root design complaint: Generate hit a completely
separate backend route (`/prompt` → `prompt_paused_model`) that never
created a diagnostic session or captured any per-node data, so Inspector
had structurally nothing to show after clicking it — not a bug to patch,
a second, disconnected code path to remove. Now there are only `>` and
`>>`, both driven through the same diagnostic-session machinery Inspector
already reads from, so the "runtime doesn't match the prompt panel"
complaint is gone by construction rather than patched over.

- `>>` now auto-starts a session itself if none exists yet (confirmed via
  AskUserQuestion) — new shared `ensureSession()` helper in
  `PausePrompt.tsx`, used by both `>` and `>>`, replacing two copies of
  near-identical start-or-continue logic. `startDiagnostic` only
  tokenizes the prompt (no sampling happens there), so no redundant
  extra step is needed before `>>` streams straight through.
- `max_new_tokens` is now a single **total budget shared across `>` and
  `>>`** within one session, not independently applied to each. Previously
  `>>` always requested a fresh `maxNewTokens` on top of whatever `>` had
  already generated — stepping twice then hitting `>>` would overshoot
  past the configured cap. Now `>>` requests `maxNewTokens - diagnosticStep`
  (the remaining budget), and both `>` and `>>` auto-close the session
  once the total reaches `maxNewTokens` — direct user spec: "once we've
  gone to the end, obviously you can't single-step again."
- Session-close no longer resets the displayed step count to 0. Real bug
  found while implementing this: the old `>>`-completion cleanup zeroed
  `diagnosticStep` immediately, which is exactly what made the Inspector's
  step counter look wrong/stale right after a completed generation — the
  final reached count now stays visible until a new prompt's first real
  step overwrites it.
- Only one output line remains — direct user request ("don't try and keep
  a history of previous prompts on the dashboard... there should only be
  one output line"). The old unlabeled second `<pre>{output}</pre>` block
  (Generate's own output, rendered with zero label directly under the
  labeled step-through box — the literal cause of the "two lines, don't
  know what either refers to" complaint) is gone along with Generate
  itself.
- Chatbot grounding preserved — confirmed via AskUserQuestion. The Lab
  Assistant's "what have you tried" context
  (`backend/chatbot/context.py::_get_prompt_history`) only ever read
  `lab.prompt` log lines, which only `/prompt` wrote. `>>`'s completion in
  `backend/api/training.py`'s `diagnostics_generate` now writes that same
  log line, so removing Generate doesn't silently blind the chatbot to
  future prompts.
- `promptModel()` removed from `frontend/src/hooks/useApi.ts` (confirmed
  unused anywhere else). The backend `/prompt` route and
  `prompt_paused_model()` themselves were left in place — deleting a
  working, self-contained, still-tested REST endpoint wasn't asked for and
  is a larger, separately-riskier change than removing the frontend button.

Tests: `test_generate_completion_logs_for_chatbot_grounding` (confirms the
new log line's exact content). Backend suite: 176 passed (was 175, +1).
Frontend: 38 passed, build clean.

**Trainer-image note:** the chatbot-logging addition touches
`backend/api/training.py` — same rebuild+repush requirement as every other
backend change this session. Everything else in this batch (config
staleness fix, PausePrompt redesign) is frontend-only.

## §47: Attention pane permanently stuck on "click > to capture" after a session closes

Direct fallout from §46, found immediately after: real bug report,
2026-07-14 — selecting the Causal Self-Attention node in Inspector kept
saying "click > to capture" even after stepping, prompting, pausing,
stepping again.

Root cause: `closeSession()` (added in §46) called
`onSessionIdChange?.(null)` whenever a session ended — including `>>`'s
now-automatic close on reaching `max_new_tokens`. App.tsx's
`peekDiagnostic` effect (the thing that refreshes attention when a
different node/head gets selected, *without* re-stepping) needs a live
`diagnosticSessionId` to target; nulling it on every close meant that once
`>>` finished (which, after §46, is most of the time — it always
auto-closes), there was no session id left for peek to use. Selecting
Causal Self-Attention *after* that point had no path to ever populate
attention data — not intermittent, structurally impossible.

Checked whether backend sessions expire and could safely be treated as
gone once "closed": they don't — `backend/training/diagnostics.py`'s
session registry (`_diagnostic_sessions`) is a plain dict with no
cleanup/TTL logic anywhere in the file. A session stays fully queryable
until the backend process itself restarts. So `closeSession()` no longer
nulls the session id — only `setDiagnosticSession(null)` (unlocks the
prompt box, PausePrompt's own local concept of "finished"). The peek
effect keeps working against the last real session indefinitely, letting
any node — including attention, with any head — be inspected retroactively
after generation completes. Starting a genuinely new prompt still
correctly overwrites the id via `ensureSession()`, so nothing stale can
leak across separate prompts.

Also removed the "No attention node selected — click a block's..." note
from the Prompt Model panel entirely — direct user request: wrong
location, duplicated Inspector's own contextual messaging, "I don't think
it does anything."

Frontend: 38 passed, build clean. Frontend-only change, no backend touched
— no rebuild needed.

## §48: "Worker stopped due to inactivity" banner firing on a cold morning load

Direct user report, 2026-07-14: starting a fresh serverless GPU run first
thing in the morning showed "This CUDA worker was stopped due to
inactivity" alongside the (correct, wanted) cold-start banner. The user
had actually stopped/deleted the endpoint manually via the Nebius console
the night before — nothing about *this* session had gone idle.

`WorkerIdleBanner.tsx` showed this message any time
`status.worker_status === "stopped"`, regardless of whether the component
had ever observed the worker in any other state. On a cold page load
where the worker was already stopped before the very first poll, the
banner fired anyway — technically not wrong (the DB row really did say
"stopped", and confirmed via grep that `idle_monitor.py` is the only place
in the backend that ever sets that status), but contextually backwards:
from the user's side this was the first action of the day, not "you went
idle mid-session." Direct user framing: fine when it's a session you were
actually using and stepped away from; wrong when you're just starting
fresh.

Fix, kept intentionally small per direct instruction ("a few lines, not
500"): a `sawReadyRef` flag, set the first time a poll observes
`worker_status === "ready"`, reset on mount/device change. The "stopped"
banner now only renders if this component actually witnessed the
ready → stopped transition itself — not on a cold load where stopped was
already the first thing polled. The warning countdown banner (unaffected)
already covers the "heads up, about to go idle" in-session case.

Tests: rewrote the old stopped-notice test to first mock a ready poll then
a stopped poll (using fake timers to advance past `POLL_INTERVAL_MS` and
trigger the real interval-driven poll, not just the initial mount poll),
confirming the banner *does* still show for a genuine witnessed
transition; added a new test confirming a cold load that only ever polls
"stopped" shows nothing. Frontend: 39 passed (was 38, net +1). Frontend-only
change, no backend touched — no rebuild needed.

## §49: Investigated "Inspector doesn't snap after Step" — not reproducible in current source

2026-07-14. User reported the Output box updates on `>` but Inspector
doesn't reflect the new snapshot immediately, feeling like it needs a
second click or a page refresh to catch up.

Static reading of the relevant code (`PausePrompt.handleStepDiagnostic` →
`onDiagnosticSnapshot` → `App.tsx`'s `setDiagnosticSnapshot` →
`Inspector`'s `diagnosticSnapshot` prop) found no lag: all four state
updates in `handleStepDiagnostic` (`setDiagnosticSnapshot`,
`setDiagnosticStep`, `setGeneratedTokens`, `onDiagnosticSnapshot`) fire
synchronously in the same tick after `stepDiagnostic` resolves — React 18
batches them into one render, so Output and Inspector can't visibly
diverge in timing.

To verify rather than just trust that reading, wrote a real integration
test suite (`frontend/src/test/pause-prompt-inspector-integration.test.tsx`)
wiring `PausePrompt` and `Inspector` together exactly the way `App.tsx`
does (same shared `diagnosticSnapshot` state, same
`onDiagnosticSnapshot` callback), with only the network layer mocked —
this is the one gap unit tests of either component alone can't catch,
since a wiring bug only exists where the two meet. Four scenarios, all
passing on the first try against current source: a generic node
(embedding) after one Step click, Runtime already open before clicking
Step, a second consecutive Step showing the second snapshot's data (not a
lagging first one), and an attention node's heatmap after one Step click
(tested separately given how much recent churn that area's had). None
reproduce the reported lag.

Conclusion: not a bug in the current source, as far as this could be
verified without a live browser. Given how extensively
`PausePrompt.tsx` has been rewritten this session (§46's Generate removal,
§47's closeSession fix), the most likely explanation is the same class of
issue hit repeatedly this session — a browser tab running a stale
Vite-HMR-patched instance of a heavily-changed component. Recommended a
hard refresh (or fresh tab) before assuming a live code bug; if it
persists after a genuinely fresh load, that's the next real signal to
chase, ideally with the exact repro sequence (which node selected, first
step vs. later step, etc.).

Frontend: 43 passed (was 39, +4 new integration tests). Build clean.
Frontend-only investigation, no source changes made — nothing to
rebuild.

## §50: Real deadlock — >/>> permanently frozen after a mixed >-then->> session hit the cap

2026-07-14, found immediately after §49's investigation (unrelated to
it) — direct user report: single-stepped a few times, then hit `>>` to
finish the rest of `max_new_tokens`. After that, typing a brand new
prompt did nothing — `>` and `>>` stayed disabled, no "Finish" button
either, no way forward short of a reload.

Root cause: `atCap = diagnosticStep >= maxNewTokens` (§46), used to
disable both buttons once the shared budget is reached.
`diagnosticStep` is deliberately left at its final value once a session
closes (§46's fix for the Inspector-step-counter-looking-stale bug) — the
*only* code path that ever resets it back to 0 is `ensureSession()`,
called from inside the very `>`/`>>` handlers `atCap` disables. Once a
session finished exactly at the cap, there was no way to ever call
`ensureSession()` again — a genuine deadlock, not just a stale display.
Fixed: `atCap` now also requires `diagnosticSession !== null` — once
`closeSession()` runs, `atCap` clears immediately regardless of what
`diagnosticStep` still says, unfreezing both buttons for the next prompt.

Verified as a real regression, not a guess: wrote
`frontend/src/components/PausePrompt.test.tsx` reproducing the exact
sequence (single-step once, `>>` to finish, type a new prompt, assert
both buttons are enabled) — confirmed it **fails** against the pre-fix
code (buttons genuinely `disabled`), then confirmed it passes with the
fix applied.

Tests: 1 new (`PausePrompt.test.tsx`). Frontend: 44 passed (was 43, +1).
Build clean. Frontend-only, no backend touched — no rebuild needed.

## §51: Paused runs permanently un-stoppable — three real bugs, all found with DB/disk evidence

2026-07-14. Direct user report: four paused runs in Open Runs, none
stoppable from the browser — 400s on some, a 502 on another. Traced each
with direct DB/filesystem inspection rather than guessing (established
pattern this session), found three distinct real bugs.

**1. Idle monitor spam / silent worker-status desync.** The terminal
message the user pasted (`Idle monitor scan failed: ... code = NotFound`)
comes from `idle_monitor.py` trying to stop a Nebius endpoint the user had
already deleted manually via the console. That exception propagated past
the DB update, so `worker_sessions.worker_status` stayed `READY` forever
— the app kept believing a long-gone endpoint was still alive, which is
exactly what let `stop_training()` try to proxy a stop call to it later.
Fixed: `stop_idle_workers()` now catches `NebiusEndpointError`, and if the
message says "NotFound" specifically, treats it as already-stopped and
proceeds to update the DB anyway. Any other CLI failure (real
network/auth issue) still re-raises — only the "it's already gone" case
is swallowed.

**2. Remote-run stop had no fallback when the proxy call fails.**
`stop_training()`'s remote branch raised a bare `HTTPException(502, ...)`
on any proxy failure and never touched the local DB — so a run whose
endpoint no longer exists (bug 1's exact scenario) stayed "paused" in Open
Runs forever, no way to clear it. Fixed: on proxy failure, mark the run
`CANCELLED` locally regardless — the remote process, if it's even still
there, is unreachable either way. Direct user request: "I should be able
to stop any of these things to kind of flush them."

**3. Local `stop_run()` required a status.json that two of the actual
stuck runs (26, 27) didn't have.** Checked their run directories directly:
`checkpoint.pt`, `metrics.jsonl`, `run_meta.json` all present, dated
2026-06-28 — but no `status.json` at all (legacy runs predating that
file's introduction). `artifacts.read_status()` returned `None` for them,
so the "process already exited, cancel it" fallback branch (which checks
`status.get("status") == PAUSED`) never matched, always falling through to
"Run not found" (400) — even though the DB itself said `paused`, which is
what the user actually sees in Open Runs. Fixed: `stop_run()` now accepts
the caller's already-fetched DB status as a fallback (`stop_training()`
already has `db_run["status"]` from its own lookup) — if there's no
status.json but the DB says paused, cancel anyway rather than requiring a
file that may simply not exist for older runs.

Also fixed the frontend silently swallowing stop failures entirely —
`OpenRunsPage.tsx`'s `handleStop` had a `try/finally` with no `catch`, so
any failure became an unhandled promise rejection visible only in the
browser console, with the run just staying in the list with zero
indication anything went wrong. Now shows a visible error banner.

Tests: `test_stop_idle_workers_treats_already_deleted_endpoint_as_stopped`,
`test_stop_idle_workers_still_raises_on_a_real_failure` (new
`tests/test_idle_monitor.py` cases), and a new `tests/test_stop_run.py`
with four cases covering the legacy-missing-status-file fallback (and its
negative case — a genuinely non-paused run still 400s), the remote-proxy-
failure fallback, and confirming the fallback doesn't mask a normally-
working remote stop. Backend suite: 182 passed (was 176, +6). Frontend: 44
passed (unchanged — no frontend logic changed, only added an error
display). Build clean.

**Trainer-image note:** none of these changes touch
`backend/training/*` or the trainer image — `idle_monitor.py`,
`backend/api/training.py`'s `/stop` route, and `runner.py`'s `stop_run()`
all run in the main server process, not the trainer container. No
rebuild needed. **A plain backend restart is required** for these fixes
to take effect (uvicorn isn't running with `--reload`) — the runs the
user was actually trying to stop (159, 27, 26) still 400/502'd against
the old running process during this same investigation, confirming the
fix wasn't live yet rather than being wrong.

## §52: Chatbot's diagnostic-snapshot tool had no size cap — blew the model's context

2026-07-14. Direct user report: second chatbot message ("comment on the
lm_head and top_k logits... I just ran a hello prompt") errored with
"error code: 400 — this model's maximum context length is 128000 tokens
... your prompt contains at least 128001 input tokens." First message
("how is it going") worked fine.

Traced the whole chatbot request pipeline (`backend/chatbot/context.py`,
`client.py`, `tools.py`) rather than guessing where the bloat came from.
`get_diagnostic_snapshot()` in `tools.py` was the one tool with no output
cap at all — every other tool in the file (`search_run_metrics`,
`search_experiment_file`) enforces `MAX_OUTPUT_CHARS = 8192`; this one
fetched the entire raw `DiagnosticSnapshot` and returned it to the model
unfiltered. Every per-position field this session added to that
schema — `position_vectors`/`input_position_vectors` on every node (up to
`DIAGNOSTIC_POSITION_WINDOW`=12 positions × n_embd floats, per node, per
input+output), `attention.qkv_detail`'s raw Q/K/V arrays — went straight
into the tool result raw. A snapshot with ~18 nodes was easily enough to
blow past 128k tokens on its own — none of those fields existed when this
tool was originally built, and it was never revisited as they were added.

Confirmed the trigger mechanism too: `client.py`'s `_LOOKUP_HINT_RE`
keyword heuristic (gates the tool-calling preflight, since streaming +
tools silently drops tool_calls on this model) matches "logit", "head",
"top_k", "embedding", etc. — exactly the words in the user's second
message, and *not* in the first ("how is it going"), which is why only
the second message triggered a tool call and hit the bug.

Also answered two related misconceptions surfaced by the same report:
runtime/trainer data (tokens, vectors, attention) was never going through
the safe grep-style tools — only `search_run_metrics`/
`search_experiment_file` are search-capped; `get_diagnostic_snapshot` is a
direct fetch, which was the actual bug. And the system prompt + README +
architecture source code are *not* sent once at conversation start —
`assemble_messages()` (`context.py`) rebuilds and resends all of that
every single turn (normal for a stateless completions API with no prompt
caching), stapled to a freshly-rebuilt volatile snapshot each time.

Fix: new `_trim_diagnostic_snapshot()` strips `position_vectors`/
`input_position_vectors` from every node and `qkv_detail` from attention
before the tool result is built — keeps shapes, summary stats, top-k
predictions, and attention weights (small, at most a 12x12 windowed
grid), matching what the system prompt already describes this tool as
providing. The tool's `note` field now tells the model to point the user
at the Inspector's Runtime tab for exact raw vector values instead of
inventing an explanation for their absence.

Tests: `test_get_diagnostic_snapshot_strips_raw_vectors_to_avoid_context_blowup`.
Backend suite: 183 passed (was 182, +1).

**Trainer-image note:** `backend/chatbot/tools.py` runs in the main
server process, not the trainer container — no rebuild needed. A plain
backend restart is required for the fix to take effect.

**Test prompt to verify the chatbot can see live trainer/runtime data**
(direct user request): step through a prompt in the Inspector first (at
least one `>` click, ideally with an attention node + head selected so
`attention` data is populated too), then ask the Lab Assistant:

> "Look at the current diagnostic snapshot — what's the output shape and
> summary stats (mean/std) for the embedding node, and what are the top-3
> next-token predictions from the LM head?"

That phrasing hits multiple `_LOOKUP_HINT_RE` keywords ("shape", "node",
"embedding", "top" via "top-3") to reliably trigger the tool-calling path,
and asks for exactly what survives the trim (shapes, summary stats,
top-k) — a good, cheap smoke test that both the tool fires and the fix
didn't remove anything genuinely useful.

## §53: "Clear chat" — reset a stuck Lab Assistant conversation without losing the experiment

Direct user request, 2026-07-14: "what if I'm in one experiment with lots
of runs and something's gone wrong with Lab Assistant and I need to reset
that?" No clear/reset mechanism existed at all — the chatbot API only had
`GET messages`, `POST message`, and the feedback PATCH.

New `DELETE /api/chatbot/{experiment_id}/messages` route + `db.py`'s
`clear_chat_messages()` — deletes only that experiment's `chat_messages`
rows, leaving the experiment, its config, and every one of its runs
completely untouched. Frontend: `useChatStream`'s `clearMessages()` calls
it and resets local state to empty (raw `fetch`, matching the rest of
that hook rather than the `api()` helper in `useApi.ts` — added a version
there first, then removed it once unused, to avoid dead code). `ChatPanel`
gained a "Clear chat" button in the header (only shown once there's
actual history), requiring a second click ("Click to confirm") before it
actually deletes anything — a real delete, so a one-click wipe felt too
easy to trigger by accident.

Separately investigated the same night: user reported saying "hello" to
the Lab Assistant got no response after restarting both backend and
frontend. Checked the live session log directly — zero request ever
reached the backend, not even a failed attempt, and no code changes this
session touched `ChatPanel.tsx`/`useChatStream.ts` at all. Matches the
stale-browser-tab pattern hit repeatedly tonight: restarting the
`npm run dev`/`uvicorn` *processes* doesn't refresh an already-open tab's
loaded JS — recommended a hard refresh/new tab rather than guessing at a
code bug that the evidence didn't support.

Tests: `test_clear_messages_deletes_history_for_experiment`,
`test_clear_messages_404_for_unknown_experiment` (backend); "shows no
Clear chat button when there's no history yet", "Clear chat requires a
second click to confirm before actually clearing" (frontend). Backend
suite: 185 passed (was 183, +2). Frontend: 46 passed (was 44, +2). Build
clean.

**Trainer-image note:** `backend/api/chatbot.py` and `backend/db.py`
changes only — chatbot never proxies to trainer containers (no reason
to; it only needs the main server's own DB/logs). No rebuild needed, a
plain backend restart is enough.

## §54: Quoted LM Head tokens, "Output Summary Stats" label

Two small clarity fixes, direct user requests, 2026-07-14.

**LM Head bar chart now quotes the token character** (`"e"` instead of
bare `e`). A space character rendered as visibly nothing before this —
the row looked empty/broken rather than a real, meaningful top-k
prediction, while punctuation like `,`/`.` was already easy to read since
those characters aren't invisible.

**Generic node Summary Stats relabeled to "Output Summary Stats."**
Verified against the actual hook code first rather than assuming: every
template's `register_diagnostic_hooks` computes
`summary = _compute_summary(output)` — confirmed always the output
tensor, never input. Previously just said "Summary Stats:" with no
indication which tensor it described, requiring a guess.

Frontend: 46 passed (unchanged — display-only text changes, nothing new
to test). Build clean. Frontend-only, no backend touched — no rebuild
needed.

## §55: Chatbot fabricated a "logging inconsistency" for a remote run — `search_run_metrics` couldn't see remote data at all

User asked the Lab Assistant for validation loss at step 500 on a
completed run. It replied that no matching records existed for step 500
and speculated this "may indicate a minor logging inconsistency,"
citing nearby steps 490/530 instead.

**Verified via direct DB query that this was false.** Run 167's
`val_loss_history` column has all 100 entries, unbroken, including step
500 exactly: `val_loss=2.0626`. There was no gap in the data anywhere
near step 500 — the model invented the "inconsistency" narrative to
explain an empty tool result it didn't understand.

**Root cause:** `search_run_metrics` (the chatbot's exact-lookup tool)
only ever read `data/runs/{run_id}/metrics.jsonl` — a file written by
`train_worker.py` for **local** runs only. Run 167 is remote
(`execution_backend='nebius_endpoint'`); `data/runs/167/` doesn't exist
on disk at all. The tool's `_search_file` correctly reported "not
found," but the query wrapper had no fallback and no way to signal
"this run's data lives somewhere else" — so the model was handed an
empty search with no explanation and filled the gap with a guess.
Same class of bug as §52 (a tool silently degrading for remote runs
instead of erroring loudly or falling back).

**Fix:** confirmed both local (`train_worker.py::write_metric()`) and
remote (`backend/api/training.py`'s `/metrics` route sync logic) paths
write every metric row into `training_runs.train_loss_history` /
`val_loss_history` (JSON columns) — this is a full-resolution, always
populated, universal source for both execution backends, unlike the
file which only exists locally. `search_run_metrics` now checks
whether `metrics.jsonl` exists per run_id; if not, it falls back to a
new `_search_remote_run_metrics()` that reads the two DB history
columns, merges rows by `step`, and searches over the same JSON-line
representation the file-based path already produces (`_search_file`
and the new DB path both now share a `_search_lines()` core so the
matching/truncation logic isn't duplicated). If a remote run somehow
has no DB-synced history either, the tool now returns an explicit
`"No synced metrics in the database for run {id}"` error instead of a
silent empty result — so the model has something honest to relay
instead of a blank slate to rationalize.

This changed `search_run_metrics` from sync to async (DB access is
async); `execute_tool_call`'s dispatch now `await`s it, matching the
existing pattern already used for `get_diagnostic_snapshot`.

Regression tests added: one reproduces the exact incident (remote run,
no local file, DB has step 500 at `val_loss=2.0626`, query "500" must
find it), one confirms the tool still reports a real error (not a
silent empty result) when a remote run has no DB history at all
either. Backend: 187 passed (was 185; +2 new tests, no regressions).
Frontend untouched. Trainer-image rebuild **not needed** — this is a
main-server-only chatbot-tool change; the chatbot itself never proxies
to the trainer container.

## §56: Five direct user reports — data-tab return location, MoE diagram honesty, hardware-info staleness, Open Runs missing a way back in, landing page decluttering

**1. Closing a data tab always landed on Inspector's Overview sub-tab, not
wherever the user actually was (almost always Runtime).** Root cause:
Inspector's sub-tab (`activeTab`) was a local `useState`, but App.tsx only
renders `<Inspector>` while `rightPaneTab === "inspector"` — opening a data
tab switches `rightPaneTab` away, which unmounts Inspector; closing it
remounts Inspector fresh, resetting `activeTab` to its default. Every other
Inspector selection (`attentionHead`, `showQKVDetail`,
`attentionWindowOffset`) was already lifted to App.tsx for exactly this
reason — `activeTab` now follows the same pattern (`Inspector.tsx`,
`App.tsx`). Since data tabs can only ever be opened from Runtime, this
naturally always lands back on Runtime without hardcoding that assumption.

**2. MoE input/output question (no code change) — confirmed correct.**
`moe/model.py::register_diagnostic_hooks` registers exactly one hook per
block, on the `MoE` module itself (`block.{i}.moe`) — not per-expert.
Captured input = `ln2(x)` (the shared input to router + all experts).
Captured output = the router-weighted, combined expert output
(`moe_out`), post-combination, pre-residual-add. No per-expert breakdown
exists anywhere in the diagnostic system — explicitly deferred ("Phase 2")
per the code's own docstring.

**3. MoE diagram showed 3 identically-wired clickable boxes for "Experts" —
misleading, since they're all the same node.** Confirmed in
`ArchSchematic.tsx`: all 3 `NodeBox`es were bound to the identical
`nodeId` (`block.{i}.moe`); clicking any of them opened identical data.
Replaced with a single clickable `NodeBox` carrying a new `segmented`
prop — 3 small internal stripes signal "multiple experts inside" without
implying 3 separately-inspectable data sources.

**4. Hardware-info dashboard fell back to CPU-only display after a
FastAPI+React restart, even for an active GPU run.** Confirmed real bug,
same class as the earlier `config`-staleness fix: App.tsx's `device` state
defaults to a hardcoded `"cpu"` and was **never re-synced from the
server** after a reload. Deeper root cause: `device` wasn't even
retrievable — it's a real `training_runs.device` column, but was absent
from every `/status` response path (local status.json, the DB fallback
`get_run_status_from_db`, the in-memory "just launched" fallback, and the
remote-proxy passthrough). Fixed by: adding `device` to `train_worker.py`'s
two `write_status()` calls, to `db.get_run_status_from_db`'s SELECT +
return dict, to `runner.py`'s in-memory fallback dict, and backfilling it
from the local DB row in `training.py`'s remote-proxy branch via
`result.setdefault("device", db_run.get("device") or "cpu")` — the
backfill means this works immediately for existing remote runs even
before a trainer-image rebuild propagates the new status.json field into
the container itself. Frontend: `RunStatus.device` added to `types.ts`;
`pollStatus` in App.tsx now calls `setDevice(status.device)` on every
poll. 2 new backend regression tests (DB fallback includes device;
remote-proxy backfill works when the proxied response omits the field).

**5a. "Open Runs" could only Stop a run, with no way back into it.**
Direct user report, confirmed in code — `OpenRunsPage.tsx` had a Stop
button and nothing else. Added an **Open** button that calls a new
`handleReopenRun()` in App.tsx: fetches the experiment's config, then
sets `experimentId`/`config`/`runId`/`device`/`backend` from the clicked
run's own fields (not defaults) so the workspace immediately resumes
polling that exact run. 1 new frontend regression test.

**5b. The landing page's unlabeled "Serverless CPU: ... (live) · Serverless
GPU: ... (live)" line was confusing and often wrong** — it always showed
both CPU and GPU as "live" regardless of what was actually running, since
it was a bare `<HardwareSpecs />` call with no run context to disambiguate
(the component's own doc comment says it shows *both* generically when no
`device`/`backend` props are given). Removed from the landing page
entirely — the labeled, per-run version inside an active workspace
(`<HardwareSpecs device={device} backend={...} />`) already shows the
correct single value and was untouched.

**5c. Preset grid too small; "Or Load an Existing Experiment" list always
inline was a distraction.** `PresetPicker.tsx` boxes enlarged ~50%
(padding 12→18, gap 10→16, name 14→17px, description 12→14px), landing
page container widened 700→900px to fit. `ExperimentBrowser` moved behind
a new "Existing Experiments" button/page (`showExperiments` state),
mirroring the existing Open Runs button/page pattern already in the app,
instead of always rendering inline on the landing view.

Backend: 189 passed (was 187; +2 new tests). Frontend: 47 passed (was 46;
+1 new test), build clean. Trainer-image rebuild recommended (not
strictly required, thanks to the backfill) for #4 — `train_worker.py`
runs inside the trainer container for remote runs, so its own
`status.json` won't include `device` natively until redeployed.

## §57: `>>` silently failing for long/MoE generations — three real bugs, one root cause plus two real gaps found chasing it

User reported `>>` (continue generation) not updating the Inspector or the
step counter on a MoE run, while the same action worked correctly on a
Tiny Transformer run. Investigation ruled out a MoE-specific bug (a direct
repro against current code, local MoE checkpoint, `>>` for 20 tokens,
worked perfectly — full node data, correct step count) and instead found
three separate, real problems stacked on top of each other:

**1. `event: error` SSE frames were silently dropped by the frontend.**
`generateDiagnosticStream` (`useApi.ts`) only ever handled `event: token`
and `event: done` — a real backend failure mid-`>>` looked identical to
success: the stream just ended, nothing updated, no error surfaced. This
alone explains "nothing happens, no explanation" for *any* underlying
failure, not just this one. Fixed: `event: error` now `throw`s, caught by
`PausePrompt.tsx`'s existing try/catch (already `alert()`s other errors —
no consumer change needed). This fix is what let the user actually *see*
finding #3 below instead of it failing invisibly.

**2. `training_runs.device` wasn't live in the running backend process.**
Confirmed via a direct DB query (`./lab.db` — note there are decoy empty
`lab.db` files at `data/lab.db` and `frontend/lab.db`, the real one is
repo-root) that the MoE run's `device` column was correctly `'cuda'` the
whole time; the bug was that the backend process (checked via `ps aux`)
had been running since before §56's device-sync backend changes landed —
Python/uvicorn doesn't hot-reload code changes without `--reload`. Not a
new code bug — the fix from §56 was correct, just not yet loaded into the
live process. No code change; documented here since it consumed real
investigation time and is a recurring class of "user reports bug, fix
already exists, backend just needs a restart" — worth remembering as the
first thing to check before assuming new code is broken.

**3. The real bug: diagnostic generation never windowed the context to
`block_size`, unlike real generation.** Both templates' own
`model.generate()` (the actual training-time generation method) correctly
slides the window every step — `idx_cond = idx[:, -self.block_size:]` —
before every forward pass. The diagnostics code path, written separately
(not sharing that method), never did this: `_execute_forward_pass()` in
`diagnostics.py` (used by `/step`, `/peek`, and `/generate`'s final-frame
capture) and `/generate`'s own inline per-token sampling loop in
`training.py` both fed the full, ever-growing `prompt_tokens +
token_history` straight into the model. Once total length exceeded
`block_size` (confirmed live: `block_size=128`, `max_new_tokens=150` —
guaranteed to cross it), RoPE's position-dependent buffers (or the causal
mask, for `pos_encoding=learned`/other configs) raised a tensor-size
mismatch — previously invisible due to bug #1, now surfaced correctly as
"Error during generation: The size of tensor a (129) must match the size
of tensor b (128)...". This is exactly the kind of split-implementation
drift the user flagged directly: *"the code for MoE and Tiny Transformer
should try and share as many methods as possible, just so to avoid this
kind of bug"* — the diagnostics path duplicated `generate()`'s sampling
loop without duplicating its windowing, and the duplication was never
template-specific in the first place (both templates hit the identical
bug, since the fix lives in shared `diagnostics.py`/`training.py` code,
not per-template code). Fixed by reassigning `all_tokens =
all_tokens[-session.model.block_size:]` right after building it in both
locations — reassigning the variable itself (not just a separate slice
fed to the model) keeps every downstream use in `_execute_forward_pass`
(`top_k_by_position`, `position_tokens`, node capture) consistent with
the same window the model actually saw.

**4. Live temperature/decoding_mode override, mid-prompting.** Direct
user request: adjust `temperature`/`decoding_mode` while paused and
partway through a diagnostic session, without restarting it (which would
lose `token_history`). `max_new_tokens` already worked this way — the
`maxNewTokens` prop reads `config.inference.max_new_tokens` live on every
render, and ConfigPanel isn't disabled while paused, so editing it there
already took effect on the next `>`/`>>` with no code change needed.
`temperature`/`decoding_mode` were not: `DiagnosticSession.temperature`/
`decoding_mode` were read once from config at `/start` and permanently
fixed for the session's life. Fixed by adding optional `temperature`/
`decoding_mode` fields to `DiagnosticsStepRequest`/
`DiagnosticsGenerateRequest`; when present, the `/step` and `/generate`
routes mutate `session.temperature`/`decoding_mode` in place before
running — the mutation persists for subsequent clicks too, not just a
one-shot override. Also fixed a latent bug found while wiring the remote
proxy body for `/step`: it was only ever populated inside the
`if attention_params is not None` branch, so *any* override (including
this new one) would have been silently dropped for remote runs whenever
attention wasn't also selected — rewritten to build the body field by
field.

**First attempt at the frontend for #4 was wrong — corrected on direct
user feedback.** Initially added a second, duplicate set of
temperature/decoding-mode controls directly under the Prompt Model box.
User immediately flagged this as redundant with the *already-existing*,
already-live-editable Inference section in ConfigPanel (left sidebar) —
correct call. Removed the new controls entirely; `PausePrompt` now just
takes `temperature`/`decodingMode` as live props sourced from
`config.inference.*`, exactly the same pattern `maxNewTokens` already
used — no new UI, no local state, single source of truth. **Lesson: when
adding a control for a setting that might already be configurable
elsewhere, check for an existing home for it before adding a new one.**

Regression tests added: SSE `event: error` now throws (frontend);
`search`... n/a; block_size windowing for both `/step` (looped past
block_size) and `/generate` (streamed past block_size) — both verified to
actually fail without the fix (reverted temporarily, confirmed the exact
same `RuntimeError` class the user's screenshot showed, restored); live
temperature/decoding_mode override for both `/step` and `/generate`,
persisting across omitted-field calls; remote-proxy body forwarding
without attention params. Backend: 194 passed (was 189; +5 new tests).
Frontend: 48 passed (was 47; +1 new test), build clean.

**Trainer-image rebuild note:** #3 (block_size windowing) and #4 (live
overrides) are both in `diagnostics.py`/`training.py` — code that runs
*inside* the trainer container for remote/serverless runs (proxied calls
execute there). Both need a trainer-image rebuild+push to take effect for
existing remote runs, unlike §56's items which were mostly main-server-only.

## §58: max_new_tokens/block_size validation cap, and a general decimal-typing bug found while adding it

User's MoE run kept crashing at `max_new_tokens=150` even after restarting
the main backend — traced to the run being **remote** (`nebius_endpoint`):
§57's fixes live in `diagnostics.py`/`training.py`, which for a remote run
execute *inside the Nebius trainer container*, not the local process a
`uvicorn` restart touches. Confirmed via the same live `ps aux` check used
in §57 (`ps` timestamp predates the code) plus the run's own
`execution_backend` column. Explained clearly to the user rather than
continuing to debug already-fixed-and-tested code.

Given the trainer-image rebuild is a separate manual step outside this
conversation's control, user asked for a defensive guard on top of the
already-fixed sliding-window generation, regardless: **reject any config
where `inference.max_new_tokens > model.block_size`**, applying identically
to every template — no template-specific code, single validation point.

Added a check to `PATCH /experiments/{id}/config` (`backend/api/
experiments.py::update_config`) — the only route that can ever change
either field — comparing `req.config["inference"]["max_new_tokens"]`
against `req.config["model"]["block_size"]` and returning `400` with a
specific message (`"max_new_tokens (150) cannot exceed block_size
(128)"`) if violated. No separate check needed at `/training/start`: every
built-in preset already satisfies the constraint (checked directly), and
this PATCH route is the only reachable way to end up violating it.

**Found and fixed two more real bugs while wiring the error through to
the UI, both were silent before this:**

1. **`handleConfigChange`'s debounced config PATCH was pure
   fire-and-forget** — no `.catch` at all. A rejected PATCH (this new
   validation, or any other 400) failed completely silently: the invalid
   value stayed shown in the UI as if it had saved, with zero indication
   anything was wrong. Fixed: `.catch()` now sets a new `configError`
   state, rendered in `ConfigPanel` the same way `TrainingControls`
   already renders `startError`/`controlError` (existing pattern, reused
   rather than inventing a new one).
2. **The shared `api()` fetch helper (`useApi.ts`) discarded the actual
   error message on every single failed request in the app** — it threw
   `"{status} {statusText}"` (e.g. `"400 Bad Request"`) without ever
   reading the response body, so FastAPI's specific `HTTPException`
   detail (the actual useful part) was always lost. Fixed generally, not
   just for this one call site: `api()` now reads `body.detail` when
   present and uses that as the error message, falling back to the
   status line only if the body isn't JSON or has no `detail` field. This
   improves every existing error message across the app for free.

**Separately, while building the decimal-typing test for ConfigPanel's
`temperature` field to verify the cap's error message rendered correctly,
found a real, general, pre-existing bug: typing a decimal into ANY
numeric config field was broken.** The plain `<input>` in
`ConfigPanel.tsx`'s `renderSection` fed `Number(e.target.value)` straight
back into the controlled `value` on every keystroke.
`Number("0.")` evaluates to `0` (a valid, non-NaN number) — so the instant
a user typed the decimal point after "0", the controlled input's
displayed value snapped back to `"0"`, permanently erasing the "." before
a second digit could ever be typed. Affected every decimal field in the
app (temperature, dropout, learning_rate, capacity_factor), not just
temperature — user reported it specifically for temperature, but it was
never temperature-specific. Fixed with a new `NumericField` component:
buffers the raw typed text in local state, decoupled from the numeric
config value on every keystroke; a `useEffect` only re-syncs the local
text from the external value when they actually diverge for a reason
other than the field's own typing (guarded by `Number(text) !== value`,
so the component's own round-trip through `onChange` never clobbers a
mid-typing state).

Regression tests added: PATCH rejects `max_new_tokens > block_size` with
both numbers named in the error message and confirms the rejected value
was never persisted; PATCH allows the boundary case (`==`) and a normal
in-range case for MoE specifically (per the "applies to every template"
requirement); ConfigPanel test simulates typing `"0."` then `"0.5"` into
the temperature field and asserts the displayed value is never clobbered,
plus the final `onChange` call carries the correct parsed number; a
second test confirms the new `error` prop renders. Backend: 197 passed
(was 194; +3 new tests). Frontend: 50 passed (was 48; +2 new tests),
build clean. Frontend-only changes for the decimal-input fix and error
surfacing — no rebuild needed; the validation cap is main-server-only
(`update_config` never proxies to the trainer) — no rebuild needed either.

## §59: LM Head "selected" highlight never showed after `>>`, temperature=0 crashed generation

**1. LM Head Inspector panel never highlighted the generated token green
after `>>` — only after single `>` steps.** Confirmed as a real, general
bug (not MoE-specific — same root cause affects every template, since the
capture logic is fully shared). `_execute_forward_pass`'s
`append_token=False` branch — used both by `>>`'s final-frame capture and
by `/peek` — always computed `lm_head.top_k` from `logits[0, -1, :]`. By
that point in the branch, `all_tokens` already ends in the just-generated
token (`>>`'s own loop appended it before calling this), so position -1
predicts what comes *next* — one token ahead of `generated_token`, which
Inspector's frontend compares `top_k` entries against to decide the
highlight. They could never match. The `append_token=True` branch (single
`>`) was correct all along: there, the forward pass runs *before* the new
token is appended, so position -1 genuinely is the distribution
`generated_token` gets sampled from.

Fixed by reusing position -2's logits (already computed in the same
forward pass, no extra cost) for the `append_token=False` case — a causal
model's output at index i is always "prediction after seeing
input[0..i]", so position T-2 (predicting T-1) is exactly the
distribution that produced `all_tokens[T-1]` (== `generated_token.id`).
Falls back to position -1 if the sequence has fewer than 2 tokens total
(no prior position to reuse).

**Testing this needed two attempts.** First test drove `/generate` over
HTTP against a real (randomly-initialized, untrained) checkpoint and
asserted `generated_token.id in top_k_ids` — it passed even with the bug
still present, reverted to confirm. Root cause: an untrained model's
argmax frequently coincides across adjacent positions purely from
initialization, not learned structure, so the test couldn't actually
discriminate fixed-vs-broken for that model/seed. Replaced with a
deterministic unit test using a fake model whose per-position argmax is
an explicit, controlled function of position index
(`token_id = position % vocab_size`) — this reliably fails without the
fix (`2 == 1` assertion, i.e. position -1's value vs the expected
position -2's value) and passes with it. **Lesson: a test built around a
real, untrained model's output can pass by chance regardless of whether
the underlying bug is fixed — prefer a small deterministic fixture when
testing "which specific tensor position got used," not "does the overall
pipeline run."**

**2. Temperature=0 crashed generation.** `torch.softmax(logits /
temperature, ...)` divides directly by `session.temperature` under
sample-mode decoding — 0 produces inf/nan, and `torch.multinomial` then
raises `"probability tensor contains either inf, nan or element < 0"`.
First implementation rejected `temperature <= 0` at `update_config`
save-time (matching the `max_new_tokens`/`block_size` pattern from §58).
**Direct user correction:** don't reject — clamp to a tiny epsilon
(`1e-6`) at the point of use instead, so 0 (or a stray negative value)
just saves fine and behaves as "as sharp as sampling allows" rather than
erroring. Replaced the `update_config` rejection with a new
`diagnostics.MIN_TEMPERATURE = 1e-6` constant, applied via
`max(session.temperature, MIN_TEMPERATURE)` at both softmax-divide call
sites (`diagnostics.py`'s `append_token=True` branch, and
`training.py`'s `/generate` inline sampling loop — same two sites §57
fixed for block_size windowing, still not sharing one implementation).
Verified both call sites actually crash without the clamp (reverted,
confirmed the exact `RuntimeError` above, restored) rather than assuming
the fix was needed.

Regression tests: `/step` and `/generate` both complete successfully
(200, no `error` SSE event) with `temperature=0` under sample decoding;
`update_config` allows saving `temperature=0` (replaces the earlier
reject-based tests from the abandoned first attempt). Backend: 201
passed (was 197; +4 new tests). Frontend: 52 passed (was 50; +2 new
tests covering `api()`'s error-detail extraction from §58, which hadn't
had direct test coverage yet), build clean.

**Trainer-image rebuild needed for both** — `diagnostics.py`/`training.py`
run inside the trainer container for remote/serverless runs, same as
§57's fixes.

**Addendum, same day: the LM Head fix above was deployed (local run,
confirmed backend process restarted after the fix) and still didn't
work.** User tested on local CPU with `>>`, no green highlight. Root
cause: fixed the wrong field. `DiagnosticSnapshot.lm_head` carries *two*
top-k fields — a flat `top_k` (single list, for the final position) and
`top_k_by_position` (one list per position, for Inspector's stepper).
`Inspector.tsx`'s `LmHeadStepper` — the component that actually renders
the LM Head panel and decides the green highlight — reads
`lm_head.top_k_by_position` exclusively and defaults to that array's
*last* entry (`isMostRecent = clampedIndex === entries.length - 1`). It
never reads the flat `top_k` field at all. The original fix only
corrected the flat field's source position; `top_k_by_position`'s loop
(`for pos in range(pos_start, T_total)`) still unconditionally ran all
the way to `T_total` regardless of branch, so its last entry was still
the same one-ahead position for the `append_token=False` case. Fixed with
the same `T-2` logic, applied to this loop's own end boundary instead
(`top_k_end = T_total if (append_token or T_total < 2) else T_total - 1`,
with a correspondingly shifted window start) — deliberately kept
independent from the `pos_start`/`T_total` pair still used by
`position_tokens`, which describes *input* tokens (correct at every
position regardless of branch, no off-by-one there) and must not shrink.

Extended the existing deterministic unit test (rather than writing a new
one) to also assert on `top_k_by_position[-1]` — confirmed it fails
without this second fix (`2 == 1`, same wrong-position signature as the
first bug) and passes with it, same revert/restore verification as
every other fix this session. **Lesson: when a bug report says a fix
"didn't work" after a verified backend restart, don't assume the fix was
wrong — check whether the frontend is even reading the field that got
fixed.** Backend: 201 passed (unchanged count — extended an existing
test rather than adding a new one). Same trainer-image rebuild
requirement as above.

## §60: LM Head highlight for every position, temperature/decoding_mode transmission verified correct, a real crash bug, and window removed

Direct user report: green "selected" highlight worked on the most recent
LM Head position, but not when browsing back to earlier positions with
◀. Separately, suspected `decoding_mode`/`temperature` weren't being
freshly read on every `>`/`>>` click.

**1. Highlight now works at every position, not just the latest.**
Previously `isSelected` required `isMostRecent` and compared against the
single `snapshot.generated_token.id` — meaningless at any position other
than the newest, since that's the only one that token describes. Backend
now computes and attaches `actual_next_token_id` to every
`top_k_by_position` entry — the real token id that occupied the next
position, reconstructed from `all_tokens` (already available server-side)
for historical positions, or from `next_token_id` for the newest position
under `append_token=True` (not yet appended to `all_tokens` at that
point). Frontend now does a plain `tk.token_id === entry.actual_next_token_id`
comparison, position-agnostic — no more `isMostRecent` special case.

**2. Live temperature/decoding_mode transmission verified correct, not
broken.** Extensive back-and-forth chasing a suspected transmission bug —
checked prop wiring (no stale closures, both call sites read fresh props
every render), wrote a passing backend test proving the mutation logic
works. User's own DevTools capture of the actual `/generate` request
body confirmed `decoding_mode: "greedy"`, `temperature: 0` were both
correctly present. The earlier "I didn't see greedy anywhere" report was
a misread (self-corrected: "that was an accident"). **The real issue was
downstream**: with correct greedy decoding confirmed, a "selected" token
showing outside top-5 or below rank #1 is mathematically impossible
(argmax is always rank 1 in its own top-k list) — so what looked like a
decoding-mode bug was actually visible fallout from the SAME
`top_k_by_position` position-mismatch bug fixed in §59/above, not a
separate transmission problem. Also clarified `max_new_tokens: 28` seen
in a request (vs. `30` in config) is correct, deliberate behavior —
total budget minus tokens already generated earlier in the same session,
not a stale/cached value.

**3. Real regression, found by the user as a runtime crash ("isMostRecent
is not defined").** While implementing #1 above, two usages of the old
`isMostRecent` variable were left behind when its declaration was
replaced — a genuine `ReferenceError` on every single `>`/`>>` click,
crashing the whole LM Head render. Both usages fixed (the highlight
condition and the "fell outside top 5" note, which now also uses
`actual_next_token_id`/looks up its display text from the adjacent
`top_k_by_position` entry rather than always `snapshot.generated_token`).
**Lesson: when replacing a variable used in multiple places, grep for
every usage before considering the edit done — a partial edit that still
type-checks (JS/TS won't catch a `ReferenceError` at build time for a
`const` used only at runtime inside a render path with no static
analysis catching it) can still crash at runtime.**

**4. LM Head position stepper's window removed entirely.** Direct user
request: "no window on that LM head runtime." Previously
`top_k_by_position` was capped to the last `DIAGNOSTIC_POSITION_WINDOW`
(12) positions, same as `position_vectors`/`qkv_detail` — but unlike
those (real per-position vectors, genuinely expensive), each
`top_k_by_position` entry is just a top-5 list of small scalars, cheap
enough to return uncapped back to the very start of the captured
sequence. `qkv_detail`/`position_vectors`/`position_tokens` remain
windowed — their cost justification is real and unchanged.

Updated an existing test whose assertions had gone stale
(`len(top_k_by_position) == DIAGNOSTIC_POSITION_WINDOW`) to assert the
opposite (every position present, from 0 to the end) — kept its
`qkv_detail`-capping assertions unchanged, since that field's window is
untouched by this change. Backend: 202 passed (was 201; net +1 after
rewriting the stale test in place rather than duplicating it). Frontend:
52 passed, build clean.

## §61: Smooth per-position window stepping for every node, not just attention

Direct user request, with a fair process complaint attached: the
attention heatmap already had a window stepper (◀/▶, shifts which
DIAGNOSTIC_POSITION_WINDOW-wide slice of history is shown), but it
stepped by the full window size (12) per click — a "discontinuous" jump,
not a smooth slide — and no other node (LayerNorm, MLP, embedding,
final_norm) had any stepper at all, only ever showing the last 12
positions with no way to see further back. Also: I'd built the backend
half of node-level windowing in an earlier response this session without
ever adding a frontend control for it, and didn't flag that gap —
confirmed directly as a real process failure ("you didn't brief me in
advance").

**Stride fixed everywhere.** Both the attention/qkv stepper and the new
node stepper now shift by 1 position per click, not by the window size.

**New per-node stepper.** `DiagnosticSession` gained `node_window_offset`
(mirrors `attention_window_offset`'s exact semantics: 0 = most recent
window, positive N = shift back N) — has to be session-level mutable
state, not a plain function argument like `attention_window_offset` is,
because node capture happens inside PyTorch forward hooks
(`register_forward_hook`, registered once at session start), which have
no mechanism to receive extra per-call arguments beyond the standard
`(module, input, output)` signature. The `/step`, `/peek`, and `/generate`
routes all mutate `session.node_window_offset` right before the forward
pass; each hook closure reads it back via `get_session(session_id)` at
capture time — same pattern already used for live temperature/
decoding_mode overrides. `_windowed_position_vectors`/`_position_vectors`
(the hook-capture helpers, one copy per template — transformer and MoE
model.py — matches the existing per-template hook duplication, not a
new inconsistency) now take an `offset` param and apply the identical
clamped-window formula attention's `window_offset` already used
(`end = max(window, T - max(offset, 0)); start = end - window`) — copied
deliberately for consistency, not reinvented.

Frontend: new `NodeWindowStepper` component, visually identical to the
attention heatmap's own stepper, rendered above `NodeVectorTable` for
any node with a `position_vectors` field. Reuses the snapshot's shared
sequence length (`generated_token.position + 1` — one forward pass, one
sequence length, applies identically to every node captured in it) as
`totalPositions`. Wired through a new `nodeWindowOffset` state in
App.tsx: reset on run/node change (same as `attentionWindowOffset`), and
a new peek effect (parallel to the existing attention one, mutually
exclusive — attention nodes use their own effect/offset, everything else
uses this one) refreshes the snapshot immediately on a stepper click
without requiring a fresh `>`.

Backend: 203 passed (was 202; +1 new test proving the offset actually
shifts `position_vectors.positions`, verified fail-without/pass-with via
the same revert/restore discipline as every other fix this session).
Frontend: 52 passed (unchanged — new component/wiring not yet covered by
a dedicated test), build clean. Needs a backend restart (session/hook
changes) to take effect; frontend-only stride/UI changes are picked up
by Vite automatically, no restart needed for those.

## §62: Small frontend polish — stepper font size, Inspector default tab, copy-to-clipboard for vectors

Three direct, small user requests, all frontend-only (no backend
restart needed for any of these):

**Stepper buttons too large.** Both the attention heatmap's stepper and
§61's new node stepper used plain, unstyled `<button>` elements inside a
`fontSize: 11` container — buttons/inputs don't inherit `font-size` from
ancestors in CSS (a common gotcha; form controls use the platform's own
UA-stylesheet font by default), so they rendered at the browser's larger
default button font regardless of the surrounding text size. Added
explicit `fontSize: 11` to all four buttons (both ◀/▶ pairs) — the
arrow glyphs shrink proportionally as a natural consequence, no separate
fix needed for "the triangle sign."

**Inspector now defaults to Runtime, not Overview.** One-line change
(`useState<SubTab>("overview")` → `useState<SubTab>("runtime")` in
App.tsx) — direct request: "that's where people are most likely to want
to go" when clicking any node. No reset-on-node-change effect exists for
this state (confirmed), so this only changes the very first tab shown
per session, not behavior when switching between already-inspected nodes.

**Copy-to-clipboard for vectors**, two places, deliberately not a
per-row copy icon in Runtime tables (discussed first — "I don't know if
that copy for the vectors... is overwhelming"): agreed a per-row icon
would duplicate the existing double-click-to-open-in-a-tab path and
clutter tables that can have many rows.
- Data tab (opened via double-click): a **Copy** button next to Close —
  copies the full-precision vector, one value per line (matches the
  tab's existing single-column layout, pastes as one spreadsheet column).
- Runtime's per-node vector tables (`VectorPreviewTable`, used for both
  Input and Output): one **Copy** button per table — copies every row at
  once as tab-separated values, one row per position (`position\tv0\tv1\t...`),
  full precision (`vectors[i]` itself, not the truncated hover-preview
  string) — pastes into a spreadsheet as a proper grid, per direct user
  preference over a flat comma-separated blob.

Frontend: 52 passed (unchanged — no new tests added for these; all three
are small, low-risk UI-only changes using the standard Clipboard API,
consistent with how trivial verified-by-inspection changes have been
handled elsewhere this session), build clean.

## §63: Nebius serverless capacity is shared and fixed, not elastic — by design

Direct user report, 2026-07-16: two concurrent runs (a MoE experiment and a
Tiny Transformer experiment), both on CPU, only ever showed **one** CPU
trainer endpoint in the Nebius console — expected two. Investigated and
confirmed this is intentional existing behavior, not a bug.

**Root cause / design, confirmed three ways:**
1. **Code** — `backend/nebius/worker_manager.py`'s own docstring: *"One
   shared endpoint per device type (session_id = 'worker-cpu' /
   'worker-gpu'), not one per user session — Track B's per-user job model
   was dropped in the 2026-07-11 pivot."* Every serverless run for a given
   device type reuses the same endpoint via `ensure_worker()`.
2. **DB** — the two runs' `training_runs` rows both had the identical
   `remote_endpoint_id`.
3. **Live logs** — `nebius ai endpoint logs` on that endpoint showed both
   runs' traffic genuinely interleaved (`GET /api/training/1/*` and
   `GET /api/training/2/*` back to back, 2-5ms responses, no contention).

**Why this works correctly, not just "happens to work":** each training
run — including inside the remote endpoint's own container, which runs the
same codebase — launches as its own OS **subprocess** (`subprocess.Popen`
in `backend/training/runner.py`), not a thread sharing one Python process.
`nebius_cpu_preset` is `16vcpu-64gb` (`config/settings.py`), so "one CPU
trainer" is a 16-core box — multiple subprocess training loops get real
multi-core parallelism, not GIL-limited interleaving.
`max_concurrent_serverless_cpu_runs` / `max_concurrent_serverless_gpu_runs`
(also `config/settings.py`) are the deliberate per-endpoint caps (default 3
each); a request beyond the cap gets a 429, it does not spin up a second
endpoint.

**Why one shared endpoint instead of one per run:** a second endpoint costs
money and takes ~2-5 minutes to cold-start (per `worker_manager.py`'s own
comments), which is wasteful when the existing endpoint has spare vCPU/VRAM
capacity — a deliberate POC trade-off from the 2026-07-11 pivot, not an
oversight.

**Current limitation, worth remembering later:** there is no hardware-tier
selector in the frontend (e.g. choosing GPU L40S vs H100, or a bigger/smaller
CPU preset) — the device picker only chooses CPU vs GPU, not a size within
either. Discussed as a future idea, not built: key `worker_manager`'s
shared-endpoint lookup by `(device_type, tier)` instead of just
`device_type` (extra `session_id` buckets like `worker-gpu-l40s` /
`worker-gpu-h100`) — same pattern already in place, just more buckets. That
gives *choice*, not *elasticity*: each tier is still one shared endpoint
capped at `max_concurrent_*_runs`; true elasticity (auto-spinning a second
endpoint of the same tier once saturated, i.e. a pool instead of one row per
device type) would be a meaningfully bigger change, only worth it if
concurrent-user demand actually exceeds 3. Explicitly deferred — not needed
for the current POC scope. See also `README.md`'s Current Status section.

## §64: Manual `>` stepping to the end never persisted a diagnostic session — only `>>` did

Direct user report, 2026-07-16, surfaced through a genuinely useful
architecture Q&A session: stepping `>` one click at a time all the way to
`maxNewTokens` reaches the exact same end state `>>` reaches in one go
(`generation_step >= maxNewTokens`, session auto-closes either way — see
the `atCap` logic in `PausePrompt.tsx`), but only `>>`'s completion ever
wrote a `diagnostic_sessions` row. A prompt run purely by manual stepping
was invisible to the Lab Assistant's `get_diagnostic_snapshot` grounding —
not a deliberate choice, just a gap: `/generate`'s completion block had
the persistence logic inlined, `/step` never called it.

**Fix**, not a `/step` behavior change — `/step`'s contract (return one
snapshot) is untouched:
- Extracted the persist logic (`>>`'s inline decode/save/prompt_log block)
  into a shared helper, `_persist_diagnostic_result(run_id, session,
  generation_params)` — reads `session.last_snapshot` (already populated
  after every `/step`/`/generate` capture) rather than taking a snapshot
  argument, so both callers can't hand it something stale.
- New route, `POST /{run_id}/diagnostics/{session_id}/finalize` — same
  local/remote `_is_remote`/`_proxy` dual-path every other diagnostics
  route uses. Requires `session.last_snapshot` to exist (400 if you call
  it before ever stepping — nothing to persist yet).
- `/generate`'s completion block now calls the same shared helper —
  behavior-identical, just de-duplicated.
- Frontend: `PausePrompt.tsx`'s existing `if (snapshot.generation_step >=
  maxNewTokens)` check (the same one that already called `closeSession()`)
  now also calls the new `finalizeDiagnosticSession()` first — the exact
  trigger condition `>>` already used, just reached via repeated `>`
  clicks instead of one `>>` burst.

Verified: 2 new tests (`tests/test_diagnostics.py`) — one confirms a
manual-stepping session reaching finalize writes exactly one
`diagnostic_sessions` row (mirroring the existing `>>` persistence test),
one confirms calling `/finalize` before any `/step` returns 400 rather
than saving a garbage row. Full suite: 205 passed (was 203).

## §65: The block-number picker was a selection dead end — changing block never touched the Inspector

**Problem (direct user report, 2026-07-14):** with an attention node selected
and a live diagnostic session, clicking a different block number in the
architecture diagram showed no change at all in the Runtime inspector — no
new heatmap, no new Q/K/V, not even the "showing stale data" warning.

**Root cause:** the numbered block buttons in `ArchSchematic.tsx` only set
the component-local `selectedBlockIdx` (which relabels the diagram and
re-targets the "Inside Block N" row). App's `selectedNodeId` — the single
source of truth for `attentionBlock`, the auto-peek effect, *and* the
Inspector's staleness warning — never changed, so nothing downstream could
react. The staleness warning couldn't even fire, because it compares the
snapshot against the same unchanged `selectedNodeId`.

**Fix:** the block buttons now also remap the current selection: if a
`block.{i}.*` child node is selected, clicking block N calls `onNodeClick`
with the same child re-addressed as `block.{N}.*`. App then updates
`selectedNodeId`, `attentionBlock` recomputes, and the existing peek effect
(§ the 2026-07-14 head-change fix) refreshes the snapshot automatically.
Nothing selected (or a non-block node selected) → picking a block behaves
as before, label-only.

Verified: 2 new tests in `frontend/src/components/ArchSchematic.test.tsx` —
remap fires with the re-indexed node id; no remap when a non-block node is
selected. Frontend suite 54 passed (was 52).

## §66: Diagnostic sessions leaked a full model each — registry never pruned, hook handles never stored

**Problem (Fable review, 2026-07-14):** every `/diagnostics/start` loads a
checkpoint into a fresh model and registers it in `_diagnostic_sessions` —
and nothing ever removed it. `delete_session()` existed but had **zero
callers**. Worse, both templates' `register_diagnostic_hooks` discarded the
`register_forward_hook` return values, so `session.hook_handles` was always
empty — even a call to `delete_session()` would have detached nothing.
Net effect: one leaked model's worth of RAM per prompt session, forever,
until process restart.

**Fix, two parts:**
- Both templates now collect and return their hook handles;
  `diagnostics.register_diagnostic_hooks` stores them on
  `session.hook_handles`, making `delete_session()`'s deregistration real.
- `create_diagnostic_session` now evicts the run's *previous* session
  (looked up via the existing `_run_to_session` map) when a new one starts
  for the same run.

**Why per-run eviction and NOT a TTL:** the frontend deliberately relies on
a "closed" session staying peekable — `PausePrompt.closeSession()` keeps the
session id alive so the Inspector can still peek attention for the final
reached state long after generation finished (see the 2026-07-14 note in
PausePrompt.tsx). The only moment that stops mattering is when a new prompt
replaces it — which is exactly when eviction now happens. A TTL would have
re-broken that flow. Remote sessions are unaffected: `_run_to_session` holds
their ids too, but they don't exist in the local registry, so
`delete_session` is a no-op for them (the leak lives in whichever process
owns the model — the trainer container has this same fix via the shared
image).

Verified: new test `test_new_session_for_same_run_evicts_previous_and_detaches_hooks`
(handles stored; eviction detaches hooks from the old model; other runs'
sessions untouched). Full suite 206 passed (was 205).

## §67: MoE attention capture always failed for every block except the first

**Problem (Fable review, 2026-07-14):** selecting Causal Self-Attention in
any MoE block other than block 1 always showed "Capture failed".

**Root cause:** `_compute_attention_weights` propagates the residual stream
through the blocks *below* the requested layer with `x = model.blocks[i](x)`.
`Block.forward` (transformer) returns a tensor, but `BlockMoe.forward`
returns `(x, drop_rate)` — so for `layer >= 1` on MoE, `x` silently became a
tuple, `B, T, C = x.shape` threw, and the function's broad `except` reported
it as "Capture failed". Layer 0 never propagates (`range(0)`), which is why
block 1 worked and masked the bug.

**Fix:** unwrap the tuple after each block call (`if isinstance(x, tuple):
x = x[0]`) — matches how `TinyMoeLM.forward` itself consumes its blocks.

Verified: new test `test_attention_recompute_works_for_moe_layer_above_zero`
— confirmed failing before the fix (revert → fail → restore → pass), passes
after. Note for remote runs: this file ships inside the trainer image, so
the fix needs a `scripts/build_push_trainer_*.sh` rebuild to take effect
serverless-side.

## §68: RoPE was never applied in the manual attention recompute — heatmap/Q/K numerically wrong for rope models

**Problem (Fable review, 2026-07-14):** `_compute_attention_weights`
replicates the fused attention path explicitly (QKᵀ → scale → mask →
softmax) because the fused path doesn't expose weights. It handled the
learned-position path (`pos_emb` added before the blocks) but never applied
`attn.rope` to Q/K — which `MultiHeadSelfAttention.forward` does. So for
any rope model (MoE's default `pos_encoding="rope"`, or a transformer
configured that way), every heatmap and Q/K vector shown in the Inspector
was the attention of a *position-blind* model — plausible-looking numbers,
just not the trained model's.

**Fix:** apply `attn.rope(q)` / `attn.rope(k)` under the same
`pos_encoding == "rope"` condition the real forward uses. V deliberately
untouched — RoPE rotates only Q/K. The Q/K vectors in `qkv_detail` now show
the *rotated* values, matching what actually enters the score computation.

**Testing note (worth remembering):** the first version of the regression
test passed even against the broken code — a freshly-initialized model's
qkv weights (std 0.02) give near-zero scores, so softmax is ~uniform with
or without RoPE and the difference sat below any tolerance. The test now
scales the qkv weight up 50x so the scores are O(1), then uses the model's
own fused forward as the oracle: reconstruct `attn(x_ln)` from the
recomputed per-head weights + V (att @ V → concat → out_proj) and require
allclose. Confirmed failing pre-fix, passing post-fix. Same trainer-image
rebuild caveat as §67 for remote runs.

## §69: False "Backend disconnected" banner — the disconnect heuristic parsed error MESSAGE text

**Problem (direct user report, 2026-07-14 evening):** after restarting both
servers and refreshing the browser, a persistent red "Backend disconnected"
banner appeared even though the backend was up. It only cleared once a new
run was started.

**Root cause — two individually-fine changes that were incompatible:**
- `api()` (useApi.ts) was improved to throw FastAPI's `detail` string
  ("Run not found") instead of "404 Not Found".
- The poll loop's disconnect heuristic (App.tsx) classified network-vs-HTTP
  by whether the error MESSAGE started with a 4xx status code
  (`err.message.match(/^4\d\d/)`).

Once detail strings arrived, every 4xx-with-body was misread as a network
failure. The observed scenario: sessionStorage restores `runId` across a
hard refresh; after the backend restart that run 4xx'd on every poll; 3
polls later the banner appeared and stayed. (A 5xx also counted as
"disconnected" — equally wrong; a 500 proves the backend answered.)

**Fix:** `api()` now throws an `ApiError` subclass carrying `status` as a
real field; the poll loop classifies with `!(err instanceof ApiError)` — any
answered request (4xx or 5xx) means connected; only genuine fetch failures
(TypeError et al.) count toward the banner. A status-as-field check can't
drift with message wording, which is exactly how the original heuristic
broke.

Verified: new test in `useApi.test.ts` (404-with-detail → ApiError,
status=404, message="Run not found"). Frontend suite 55 passed.

## §70: Startup reconciliation marked live remote runs as failed

**Problem (Fable review, 2026-07-14):** `reconcile_orphaned_runs()` (run on
every startup from `main.py`'s lifespan) marked EVERY active-status run
failed with "Backend restarted — worker lost". Correct for local runs — the
worker subprocess dies with the parent (pdeathsig). Wrong for remote runs:
the Nebius trainer container survives a local API restart and keeps
training (and billing). The failure was masked in practice because the
`/status` route re-syncs the local row from the live remote status on the
next frontend poll — but with no browser open, the run stayed "failed"
locally while the endpoint kept working.

**Fix:** the UPDATE now exempts remote runs that actually have something
remote: `AND (execution_backend IS NULL OR execution_backend = 'local' OR
remote_run_id IS NULL)`. The `remote_run_id IS NULL` clause deliberately
keeps one remote case reconcilable — a run whose provisioning task
(in-memory `_provisioning_tasks`) died with the restart before mirroring
anything to the endpoint; nothing remote exists for it, so it *is*
orphaned. `IS NULL` on execution_backend covers pre-migration rows.

Verified: new `tests/test_reconcile.py` (local running → failed; remote
running with remote_run_id → untouched; remote mid-provisioning → failed;
local paused → untouched). Confirmed failing pre-fix. Full suite 209 passed.

## §71: Sync torch in async routes froze the whole API — moved to worker threads, with a diagnostics lock

**Problem (Fable review, 2026-07-15):** every diagnostics route is `async
def` but ran synchronous torch directly on the event loop: forward passes
in `/step`/`/peek`, checkpoint loads in `/diagnostics/start`,
`/architecture/embedding-table` and `/prompt`, a model build in
`/architecture`, and — worst — the `>>` SSE loop, which blocked the loop
once per generated token. During any of these, *every* other request
stalled, including the frontend's 2s status poll (which pre-§69 could then
even trip the false disconnected banner).

**Fix:** the blocking sections now run via `asyncio.to_thread(...)`. The
`>>` loop offloads one token at a time, so frames still stream as they're
produced.

**The subtle part — `_diag_lock`:** the event loop's blocking behavior was
also an accidental *serialization guarantee* — no two session-touching
calls could ever interleave mid-mutation. Threads remove that guarantee,
and the frontend genuinely fires overlapping requests (the Inspector's
peek effect triggers while a step is in flight). A single global
`asyncio.Lock` in training.py now wraps every session-mutating torch call
(`start`/`step`/`peek`/each `>>` token/final capture). Global rather than
per-session: after §66 sessions are one-per-run and this is a single-user
lab — per-session locks would add bookkeeping for no observable gain. The
`>>` loop takes the lock per token (not across the stream), preserving the
old behavior of peeks interleaving between tokens. Session-free loads
(`/prompt`, embedding table, param count) use `to_thread` without the lock.

Also fixed in passing: an `HTTPException` raised inside `/diagnostics/
start`'s try-block (e.g. 400 "Unknown template") was previously swallowed
by the broad `except Exception` and re-raised as a 500 — it now passes
through with its real status.

Verified: new test `test_slow_diagnostic_step_does_not_block_event_loop`
(a 1.5s-slow step must not delay `/api/health`). First version passed even
against blocking code — the health request won the initial scheduling race
— so the test now gives the step a 0.3s head start before timing; confirmed
failing pre-fix after that. Full suite 210 passed.

## §72: Remote diagnostic persistence was container-local — the controller now mirrors it

**Problem (Fable review, 2026-07-15):** for remote runs, `/generate` and
`/finalize` are proxied wholesale to the trainer container, so
`_persist_diagnostic_result` ran *inside the container* — writing that
container's own SQLite and prompt log, both of which die with the
container. The Lab Assistant is grounded on the CONTROLLER's lab.db and
logs, so serverless prompt history was simply invisible to it (same class
of gap as §16/§17: local row never updated for remote activity).

**Fix:**
- `_persist_diagnostic_result` now returns the payload it persisted
  (prompt, generated_output, generation_params, top_k_summary), and the
  local `/finalize` includes it in the response as `persisted` (extra key,
  invisible to the frontend which only reads `success`).
- New `_persist_remote_diagnostic()` on the controller mirrors that payload
  into the local `diagnostic_sessions` table + prompt log, tagged
  `(remote)` in the log line.
- Controller's `/finalize` remote branch mirrors after proxying.
- Controller's `/generate` remote branch calls the remote `/finalize` after
  the stream completes and mirrors its payload. Deliberate trade-off: this
  writes a second, identical row in the *container's* throwaway DB (its
  /generate already persisted once) — nothing ever reads that DB, and the
  alternative was a separate payload-only route on the trainer. Mirror
  failures log a warning and never corrupt the already-delivered stream.

**Stale-image tolerance:** a trainer image built before this change returns
plain `{"success": true}` — the controller logs "trainer image predates
§72?" and skips the mirror rather than erroring. As with §67/§68, the full
fix needs a trainer-image rebuild to work end-to-end serverless.

Verified: `test_remote_finalize_mirrors_persistence_into_local_db`
(confirmed failing pre-fix) and
`test_remote_finalize_tolerates_stale_trainer_image`. Existing local
finalize test updated for the new `persisted` response key. Full suite
212 passed.

## §73: Fresh runs trained max_iters + 1 steps — loop started at 0 with an inclusive end

**Problem (Fable review, 2026-07-15):** `train_transformer`/`train_moe`
looped `range(start_step, max_iters + 1)` with `start_step = 0` on a fresh
run — max_iters + 1 optimizer steps, i.e. "500 of 500" had actually trained
501 times. Checked for an existing rationale before changing (per direct
user instruction): the only nearby comments explain *resume* semantics
("checkpoint saved AFTER completing step N, so resume from N+1") and
pause-check placement — nothing explains the extra fresh-run step, and the
resume arithmetic itself assumes "step N = N completed steps", which a
0-start contradicts.

**Fix:** fresh runs start at step 1 (`else 1`); the inclusive `max_iters +
1` end is kept so the final step is numbered max_iters and eval/checkpoint
still fire on it. Resume path untouched. RNN's loop already counted from 1
(`counter += 1` before use) — no change there. The now-redundant
`step > 0` eval guard is left in place (harmless, and removing working code
isn't worth the regression surface).

Verified: new `tests/test_train_loop_steps.py` runs a real tiny
train_transformer with a counting optimizer (dataset/DB monkeypatched,
everything under tmp_path) — exactly max_iters steps, current_step still
ends at max_iters. Confirmed failing pre-fix. Full suite 213 passed.

## §74: Demo fixtures moved out of the main bundle via dynamic import

**Problem (Fable review, 2026-07-15):** ~230 lines of demo fixture data
(`FIXTURE_MANIFEST`, `FIXTURE_SNAPSHOT_WITH_ATTENTION`, `FIXTURE_SNAPSHOT`,
plus the inline `/diagnostics/start` response) lived at module scope in
`useApi.ts`. Every real user's browser downloaded them even though they are
only reachable behind `?use_fixtures=true`.

**Fix:** fixtures now live in `frontend/src/fixtures/diagnostics.ts` and are
loaded with `await import("../fixtures/diagnostics")` *inside* each
`useFixtures()` branch. The dynamic import is the load-bearing part: a
static top-level import would re-create the same bloat under a new filename,
whereas dynamic import makes Vite emit a separate chunk
(`dist/assets/diagnostics-*.js`, ~5.8 kB) that only demo sessions fetch.
Five consumers (`fetchArchitecture`, `startDiagnostic`, `stepDiagnostic`,
`peekDiagnostic`, `getDiagnosticSession`) became `async function` — callers
are unaffected since they already returned Promises.
`generateDiagnosticStream` was already an async generator.

**Do not** convert the dynamic imports back to static ones during refactors
— tree-shaking will NOT remove the fixtures (they're referenced), and the
demo data would silently return to the production bundle.

Verified: post-build, `grep -c "diag-17" dist/assets/index-*.js` → 0 (was 1)
and the marker appears only in the diagnostics chunk. Frontend suite 55
passed, build clean.

## §75: DRY refactor pass — five duplications extracted (2026-07-15)

Behavior-preserving refactors of the duplications catalogued in
docs/Fable Codebase Review.md ("Code bloat / DRY"). One commit each, full
suites (backend 213 / frontend 55 + build) green after every one. Things
future refactors must not regress:

- **WindowStepper** (`df6a39b`): `frontend/src/components/WindowStepper.tsx`
  replaces `NodeWindowStepper` and the inline stepper in `AttentionHeatmap`
  (both in Inspector.tsx). Hide conditions stay at the call sites.
- **CopyIconButton** (`1f9d00a`): one component, `size` prop (12 Inspector /
  14 App+ChatPanel), two modes — uncontrolled (self-managed 1.5 s `copied`
  state) for Inspector/App, controlled (`copied` + `onCopied` props) for
  ChatPanel, whose per-message `copiedId` keying must stay in ChatPanel so
  the checkmark shows only on the copied message.
- **Diagnostic hook factory** (`4c6d8f6`): shared `make_hook_for_diagnostics`
  + `_get_position_vectors` in `backend/training/diagnostics.py`; both model
  templates' `register_diagnostic_hooks` are now thin registration loops.
  The factory is tuple-tolerant (MoE blocks emit `(tensor, drop_rate)`).
  Two invariants: templates still RETURN the collected hook handles (§66
  eviction depends on it), and the import of the factory is deferred inside
  the method — hoisting it to module top level risks a circular import
  (diagnostics.py itself loads models from the templates).
- **`_train_char_lm` merge** (`92419cb`): `train_transformer`/`train_moe` are
  wrappers around one loop parameterized by registry key, `eval_fn`, and
  `build_metric_row` (MoE adds drop_rate fields). Forward-output arity is
  checked via `len()` at runtime. The §73 `start_step = … else 1` fix and
  comment are preserved. `OPTIMIZERS`/`load_tiny_shakespeare`/
  `sync_update_training_run` must stay looked-up at CALL time, not captured
  at import — `tests/test_train_loop_steps.py` monkeypatches them.
- **Test fakes** (`7dcb5e5`): `FakeResponse`/`FakeAsyncClient` live once in
  `tests/conftest.py`, imported by test_training_remote.py and
  test_diagnostics.py.

Remote note: `4c6d8f6` and `92419cb` touch `backend/training/` — they join
§67/§68/§71/§72 on the list of changes that need a trainer-image rebuild
(`scripts/build_push_trainer_*.sh`) before they reach serverless runs.
Deliberately still deferred: db.py connection boilerplate (review item 7).

## §76: Attention captured eagerly for ALL layers × heads (attention_maps)

**Problem (user report, 2026-07-15):** attention was the only node type
captured on-demand — the snapshot held weights for just the last (layer,
head) pair the UI requested. Two consequences: switching block/head cost a
0.9–2.8 s peek that grew with layer depth (the recompute walks the model up
to the selected layer), and the Lab Assistant couldn't see attention at all
unless the user had manually clicked a block + attention node first (its
tool just reads the snapshot).

**Fix (commits a542eae backend, 36dda25 frontend, spec'd by Fable,
implemented by a Sonnet subagent over two review rounds):** every step/peek
computes `snapshot["attention_maps"]` — windowed (12-position) softmax
weights for all n_layer × n_head pairs, ~25–35 KB at preset size, floats
rounded to 4 dp. Cost is ONE manual propagation (all heads per layer come
from the same fused QKV), about the same as one worst-case single-pair peek
— and it eliminates the per-click peeks entirely.

Key invariants:
- `_layer_qkv(block, x)` is the single home of the ln1/QKV/RoPE core; both
  `_compute_all_attention` and `_compute_attention_weights` call it. The
  §67 tuple unwrap appears only in the two propagation loops. Correctness
  chain: §68's fused-forward oracle anchors the single-pair path to the
  model's real output; the new agreement-oracle test pins
  attention_maps[layer][head] to the single-pair path (mutation-verified:
  advancing x before capture makes it fail).
- `_compute_all_attention` must stay ONE pass. The subagent's first draft
  called a per-layer helper that re-propagated from the embedding each time
  — O(n_layer²), ~7.5 s per step. Review caught it; don't reintroduce.
- `snapshot["attention"]` (selected pair) and on-demand `qkv_detail` are
  unchanged — remote runs on a pre-§76 trainer image lack attention_maps,
  and the frontend falls back to the old peek path in that case.
- App.tsx skips the block/head-change peek only when maps are present AND
  qkv_detail is closed AND window offset is 0; any of those falsy → peek
  fires as before.
- Chatbot `_trim_diagnostic_snapshot` passes attention_maps through, so the
  assistant sees attention with zero UI interaction.

Trainer image rebuild required for remote runs (joins the §67/§68/§71/§72/
§75 rebuild list); the fallback keeps stale-image remotes functional.

## File Layout

See `README.md` for project structure and setup instructions.
