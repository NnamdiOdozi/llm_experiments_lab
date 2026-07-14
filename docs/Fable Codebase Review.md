# Fable Codebase Review

Deep review of the full codebase (~9k lines read: all major backend — `training.py`, `diagnostics.py`, `runner.py`, `train_worker.py`, `db.py`, `worker_manager.py`, `endpoints_client.py`, `idle_monitor.py`, `experiments.py`, `main.py`, both model templates, `settings.py` — and frontend — `App.tsx`, `Inspector.tsx`, `PausePrompt.tsx`, `ArchSchematic.tsx`, `useApi.ts`). Findings ordered worst first.

> **Status update 2026-07-15:** nine findings fixed, one commit each, all with regression tests and DESIGN_DECISIONS entries §65–§73: block picker dead end (`f28469b`), session memory leak (`5c9dcaa`), MoE tuple bug (`8917f4a`), RoPE recompute (`9e0eb0c`), disconnected-banner heuristic (`b571311`), reconcile filter (`f9af221`), event-loop blocking (`57827a9`), remote diagnostic persistence (`544914f`), train-loop off-by-one (`3bf45ad`). Trainer image NOT yet rebuilt — the diagnostics-side fixes don't reach remote runs until `scripts/build_push_trainer_*.sh` runs. Still open: the DRY/bloat items below and the db.py connection handling (both deferred deliberately).

---

## Reported bug 1 — changing block shows no change in the Runtime inspector. Three stacked causes.

### 1a. Block picker is a dead end (the one actually hit)

`ArchSchematic.tsx:187` — the numbered block buttons call only `setSelectedBlockIdx(idx)`, local state. They never call `onNodeClick`, so App's `selectedNodeId` stays `block.0.attention`. Everything downstream keys off `selectedNodeId`: `attentionBlock` (`App.tsx:185-186`), the peek effect (`App.tsx:230`), and even Inspector's "showing stale data" warning (`Inspector.tsx:1048-1052`). So changing block updates the little diagram label and *silently nothing else* — no refresh, no stale banner.

**Fix shape:** when `selectedBlockIdx` changes and a `block.{i}.*` child is selected, remap the selected node id to the new index (one small effect or callback in ArchSchematic/App).

### 1b. MoE: attention capture fails for every block except block 1

`diagnostics.py:294-295` propagates through earlier blocks with `x = model.blocks[i](x)`. Works for transformer (`Block.forward` returns a tensor) but `BlockMoe.forward` returns `(x, drop_rate)` (`moe/model.py:120-124`) — so for `layer ≥ 1`, `x` becomes a tuple, `B, T, C = x.shape` throws, the broad `except` at `diagnostics.py:341` swallows it, and the UI shows "Capture failed". Even after fixing 1a, MoE blocks 2–4 will never render attention until this is fixed.

### 1c. RoPE models: the displayed attention is mathematically wrong

The real forward pass applies rotary embeddings to Q and K (`transformer/model.py:72-74`). The manual recompute in `_compute_attention_weights` handles learned `pos_emb` (`diagnostics.py:292-293`) but never applies `attn.rope` to q/k. MoE defaults to `pos_encoding="rope"` (`moe/model.py:139`), so every MoE heatmap and Q/K vector shown so far is the attention of a *position-blind* model, not the trained one. Same for any transformer configured with rope. Subtle and educationally the most damaging — the whole point of the panel is "what the model actually computed".

---

## Reported bug 2 — false "Backend disconnected" banner

Root cause: two changes that are individually fine but incompatible.

- `useApi.ts:19-26` (recent improvement) throws FastAPI's `detail` string ("Run not found") instead of "404 Not Found".
- `App.tsx:391` classifies HTTP-vs-network by `!err.message.match(/^4\d\d/)` — i.e. it assumes the message *starts with the status code*.

So any 4xx whose body has a `detail` field is now counted as a network failure; after 3 polls (`App.tsx:392-394`) the red banner appears and stays as long as polls keep 4xx-ing. The evening scenario fits exactly: sessionStorage restored an old `runId` across the refresh (`App.tsx:36-52` — sessionStorage survives hard refresh), the restarted backend answered 404/400 with a detail body, and the banner sat there until a new run started (new runId → polls succeed → banner cleared). The backend was never disconnected.

**Fix shape:** make `api()` throw a custom error carrying `res.status` as a field, and classify on that field, not on message text. (Side note: a 5xx also currently counts as "disconnected", which is wrong — a 500 proves the backend is very much connected.)

**Related:** `main.py` lifespan calls `reconcile_orphaned_runs()` on every startup, which marks **all** active runs failed — that's why the training "disappeared" after the restart. For local runs that's correct (the worker died with the parent). But `db.py:449-464` doesn't filter by `execution_backend`, so a *remote* run still happily training in the Nebius container also gets marked failed locally. It self-heals only if the frontend keeps polling `/status` (which re-syncs from remote); otherwise the DB says failed while the endpoint burns money. Should be `WHERE ... AND execution_backend = 'local'`.

---

## Other significant findings

### Memory leak — diagnostic sessions are immortal

`_diagnostic_sessions` is never pruned; `delete_session` exists but has **zero callers** (grepped). Every `/diagnostics/start` loads a full model checkpoint into RAM and it stays there until process restart. Worse, `hook_handles` is never populated — both `register_diagnostic_hooks` implementations discard the `register_forward_hook` return values (`transformer/model.py:258-269`, `moe/model.py:285-296`) — so even calling `delete_session` wouldn't detach hooks.

**Caution for the fix:** `PausePrompt.tsx:159-175` *deliberately relies* on sessions never expiring (post-close peek). The right fix is per-run eviction — when a new session starts for a run, delete that run's previous one via the existing `_run_to_session` map — not a TTL.

### Event-loop blocking

All diagnostics routes are `async def` but run synchronous torch inside them: full forward passes in step/peek, `torch.load` of checkpoints in `/diagnostics/start` and `/architecture/embedding-table`, and the `/generate` SSE loop does a blocking forward per token. Every one of these freezes the entire API (including the 2s status poll — which, combined with bug 2's heuristic, can itself trip the disconnect banner during a slow generate). POC-acceptable, but `run_in_executor` or plain `def` routes (FastAPI threadpools those) would fix it cheaply.

### Remote diagnostic persistence is ephemeral

For remote runs, `/generate` and `/finalize` are proxied, so `_persist_diagnostic_result` runs inside the trainer container — writing that container's own SQLite and prompt log, which die with the container. The Lab Assistant grounded on the controller's `lab.db`/logs will never see remote prompt history. Same class of gap as §16/§17 in DESIGN_DECISIONS.md.

### Training off-by-one

`train_worker.py:322` — `range(start_step, max_iters + 1)` with `start_step = 0` runs `max_iters + 1` steps on a fresh run. Harmless but the step counter reads "500 of 500" after 501 optimizer steps.

### db.py robustness

- Connection-per-call with no `try/finally` — an exception between `get_db()` and `close()` leaks the connection.
- The `_MIGRATIONS` loop's bare `except: pass` swallows *any* error (locked DB, disk full), not just duplicate-column.
- `update_training_run` / `update_worker_session` / `sync_update_training_run` interpolate kwarg names into SQL f-strings — safe with current internal callers, but a fragile pattern worth an allowlist assert.

---

## Code bloat / DRY (deferred — detail for when it's picked up)

### 1. Diagnostic hooks duplicated across model templates (~120 lines → ~40)

`TinyTransformerLM.register_diagnostic_hooks` (`transformer/model.py:192-270`) and `TinyMoeLM.register_diagnostic_hooks` (`moe/model.py:219-297`) are near-verbatim copies.

**Identical in both:** the windowed-vector helper (same body, even the same `DIAGNOSTIC_POSITION_WINDOW` windowing math — only the *name* differs: `_windowed_position_vectors` vs `_position_vectors`); the `make_hook(node_id)` closure factory (get_session lookup, shape/summary/position-vector capture, NodeCapture construction); the registration sequence (embedding → per-block ln1/attention/ln2/FFN-or-MoE → final_norm → lm_head); the handle collection added in §66.

**Actually different:** (a) the MoE hook's output handling unwraps the `(x, drop_rate)` tuple before shape/summary/vectors — the transformer hook assumes a bare tensor; (b) the 4th block child: `block.{i}.mlp` hooked on `block.ffn` vs `block.{i}.moe` hooked on `block.moe`.

**Refactor shape:** move the helper + `make_hook` into `diagnostics.py`. The MoE hook is a strict superset (its tuple-tolerant branch handles the tensor case too), so both templates can share it as-is. Each model's method shrinks to its registration list — the only genuinely template-specific part. Do the transformer and MoE test files still pass unchanged afterwards? They should; that's the acceptance check.

### 2. `train_transformer` vs `train_moe` (`train_worker.py:291-351` / `374-435`, ~90% identical)

**Identical:** STARTING status, tiny_shakespeare + CharDataset setup, TEMPLATE_REGISTRY build + optimizer setup, resume/checkpoint load, `sync_metadata`, seed, the whole step loop skeleton (batch → forward → backward → step → yield_gpu → progress → pause check), eval cadence, final checkpoint + COMPLETED block.

**Actually different:** (a) registry key `"transformer"` vs `"moe"`; (b) forward unpack `_, loss = model(x, y)` vs `_, loss, _ = model(x, y)`; (c) eval helper — `_transformer_eval` returns `{split: loss}`, `_moe_eval` returns `{split: {loss, drop_rate}}`; (d) the metric row — MoE adds `train_drop_rate`/`val_drop_rate` (×100, rounded 1dp).

**Refactor shape:** one `_train_char_lm(ws, template_key, eval_fn, build_metric_row)` — the two public functions become 3-line wrappers. `_transformer_eval`/`_moe_eval` are themselves ~80% identical and could merge behind a "returns extra drop_rate field" flag, but that's second-order. Note `tests/test_train_loop_steps.py` (§73) counts optimizer steps through `train_transformer` — it must keep passing.

### 3. `useApi.ts` fixtures in the production bundle (~230 of 549 lines)

`FIXTURE_MANIFEST` (~50 lines), `FIXTURE_SNAPSHOT_WITH_ATTENTION` (~110), `FIXTURE_SNAPSHOT` (~65) live in the main API module, shipped to every user, used only behind `?use_fixtures=true`. Move to e.g. `src/fixtures/diagnostics.ts`; if bundle size matters, gate behind a dynamic `import()` so they tree-shake/lazy-load. The dozen `if (useFixtures())` branches stay — only the data moves.

### 4. Copy-icon button defined three times

Same copy/checkmark SVG glyph pair + "copied for 1.5s" logic in: `Inspector.tsx` `CopyIconButton` (12px, local `copied` bool), `App.tsx` `CopyIconButton` (14px, identical code otherwise), `ChatPanel.tsx` (inline per-message variant keyed on `copiedId: number | null` rather than a bool — the only structural difference). Extract `components/CopyIconButton.tsx` with a `size` prop; ChatPanel keeps its own keying but reuses the two SVGs (or the whole button with a `copied` prop lifted).

### 5. Window-stepper JSX duplicated inside `Inspector.tsx`

`NodeWindowStepper` (~40 lines) and the inline stepper in `AttentionHeatmap` (~30 lines) are the same UI: ◀ Earlier / "Positions X–Y of Z" / Later ▶, same `maxOffset = max(0, total - windowSize)` math, same disabled logic. Differences are only where the numbers come from (`pv.positions` vs `att.window_start`/`att.total_positions`) and the hide condition. Extract one `WindowStepper({ windowStart, windowSize, totalPositions, offset, onOffsetChange })`; both call sites shrink to one line. `VectorPreviewTable` in the same file is the pattern to imitate — that one was extracted properly.

### 6. Test doubles duplicated

`FakeResponse`/`FakeAsyncClient` are defined in `tests/test_training_remote.py` AND redefined at the top of `tests/test_diagnostics.py` — which *also* imports the other file's copies in some tests (`from tests.test_training_remote import FakeAsyncClient, FakeResponse`). Two sources of truth, already interleaved. Move to a `tests/conftest.py` or `tests/fakes.py`.

### 7. db.py connection boilerplate (explicitly deferred — volume, not a bug)

Every one of ~20 functions repeats the same 5 lines: `db = await get_db()` → execute → commit → `close()`, with no try/finally (an exception between open and close leaks the connection). One `@asynccontextmanager async def _db()` used as `async with _db() as conn:` removes ~80 lines and fixes the leak-on-exception in the same stroke. Do this as one mechanical pass; it touches every db.py function, so don't mix it with behavior changes.

### Smaller singles

- The metrics WebSocket (`training.py`, `/{run_id}/ws`, ~90 lines) is fully built and never used by the frontend (polling does the same job) — dead code; keep or delete deliberately, not by accident.
- `settings.py:76` — `gpu_idle_timeout_seconds: 1800` with comment "temporarily changed to 30mins for hackathon but should be 10 min". Flag before open-sourcing.
- `database_path: Path("lab.db")` is CWD-relative — start uvicorn from a different directory and you silently get a second empty DB. Anchor it to the project root like `data_dir` usage elsewhere.

---

## What's genuinely good

The local/remote dual-path pattern is applied consistently across all ~12 routes; status writes are atomic (`tmp` + `replace`); the worker self-heal logic (adopt/restart/verify-before-trust) clearly encodes real incidents; and the DESIGN_DECISIONS discipline (65 sections) is why 1a-style bugs are findable at all. The heavy inline comment narration is a style call — much of it could shrink to `§NN` references now that the doc exists — but it made this review faster, not slower.

---

## Suggested fix order

1. **Bug 2** (banner heuristic) — small, user-visible
2. **1a** (block picker)
3. **1b** (MoE tuple)
4. **1c** (RoPE)
5. **reconcile filter** (`execution_backend='local'`)
6. **session eviction** (per-run, via `_run_to_session`)

Note: 1b and 1c live in `backend/training/diagnostics.py`, so they need the trainer image rebuilt (`scripts/build_push_trainer_*.sh`) for remote runs to pick them up — local uvicorn restart alone is not enough.

---

*Review completed: 2026-07-14 23:29 BST (Claude Fable 5)*
*Status update + DRY detail expanded: 2026-07-15 (Claude Fable 5) — 9 findings fixed (§65–§73), DRY/db.py items deferred with implementation notes above.*
