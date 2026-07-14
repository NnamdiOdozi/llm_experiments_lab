# Fable Codebase Review

Deep review of the full codebase (~9k lines read: all major backend — `training.py`, `diagnostics.py`, `runner.py`, `train_worker.py`, `db.py`, `worker_manager.py`, `endpoints_client.py`, `idle_monitor.py`, `experiments.py`, `main.py`, both model templates, `settings.py` — and frontend — `App.tsx`, `Inspector.tsx`, `PausePrompt.tsx`, `ArchSchematic.tsx`, `useApi.ts`). **No code was changed** — review only. Findings ordered worst first.

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

## Code bloat / DRY

- `make_hook` + `_windowed_position_vectors` duplicated near-verbatim (~60 lines) between `transformer/model.py` and `moe/model.py` — extract to `diagnostics.py`; only the MoE tuple-unwrap differs.
- `train_transformer` and `train_moe` in `train_worker.py` are ~90% identical (~60 duplicated lines).
- `useApi.ts` carries ~230 lines of hard-coded fixtures in the production bundle — belongs in a test/fixtures module.
- `CopyIconButton` defined three times (App.tsx, Inspector.tsx, ChatPanel per its own comments); the window-stepper JSX duplicated inside Inspector (`NodeWindowStepper` vs the inline heatmap stepper).
- The metrics WebSocket (`training.py:602-689`) is fully built and never used by the frontend (polling does the same job) — dead code; keep or delete deliberately, not by accident.
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
