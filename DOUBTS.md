# Doubts

Judgement calls made without user sign-off, so they can be reviewed rather than
silently inherited. Newest first.

---

## 6. 2026-09-16 21:xx — Work committed to a BRANCH, not to `main`

**What:** all of today's work is on `nanogpt-export-and-panel-fixes`, not `main`.

**Why:** the user said "commit before you start, and commit after every major
feature" and then went away. Their history is entirely on `main` with no
branches, so this deviates from their habit. I branched because pushing an
unreviewed ~200-line refactor of a working export path onto `main` while they
cannot look at it seemed worse than the mild surprise of a branch.

**Affects:** nothing functional. `git checkout main && git merge
nanogpt-export-and-panel-fixes` lands it. If they would rather it had gone
straight to `main`, that merge is the whole fix.

---

## 5. 2026-09-16 — MoE export omits the drop-rate diagnostic

**What:** the lab's `MoE.forward` returns `(output, drop_rate)`. The exported
standalone script's copy returns only the tensor.

**Why:** a tuple return cannot sit inside `nn.Sequential`, which is how the
exported model composes its blocks. Carrying the drop rate out would mean
rewriting `Block.forward` and the model's forward in the exported script, for a
metric the exported script never prints.

**Affects:** routing, top-k weighting and capacity-based dropping are identical,
so training behaviour matches. Only the observability metric is missing. Called
out in a docstring inside the generated script itself, so whoever reads it knows.

**Risk if wrong:** someone uses the exported script expecting drop-rate telemetry
and finds none. Low.

---

## 4. 2026-09-16 — nanoGPT scaled init matches parameters by NAME SUFFIX

**What:** the exported nanoGPT applies GPT-2 scaled init by matching parameter
names ending `out_proj.weight` or `net.2.weight`.

**Why:** the exported model is assembled from plain modules with no marker for
"this is a residual projection". `net.2` is the second `Linear` inside
`FeedForward`'s `nn.Sequential(Linear, GELU, Linear, Dropout)`.

**Affects:** **brittle.** If anyone reorders that `Sequential`, `net.2` silently
stops being the projection and the init quietly stops applying — no error. The
lab's own model doesn't have this problem; only the exported copy does.

**Risk if wrong:** slightly different initialisation in exported runs. Would not
crash, and would be hard to notice. A named submodule would be sturdier.

---

## 3. 2026-09-16 — kept the parameter count on the landing-page card

**What:** the user listed things to cut and said "the parameter count, we can do
that", which could mean keep or cut. I kept it.

**Why:** their other bullets said "take that out" explicitly, and 124M vs 1.8M is
the one fact that explains why the preset exists at all.

**Affects:** one clause of user-facing copy. Trivially reversible. Flagged to them
at the time.

---

## 2. 2026-09-16 — the RoPE newline fix changes previously-broken output

**What:** fixed `\\n` → `\n` in the attention RoPE branch of the export builder.

**Why:** it emitted a literal backslash-n, making any rope-configured export
invalid Python (`SyntaxError`).

**Affects:** no existing golden changed, because transformer uses learned
positions and MoE was never exportable — so nothing was relying on the broken
output. But if any user ever downloaded a rope export, their file was broken and
now would not be. Strictly an improvement; recorded because it is a silent
behaviour change to a code path nobody had tested.

---

## 1. 2026-09-16 — `lab.db` was edited directly to clear a stale endpoint URL

**What:** ran `UPDATE worker_sessions SET endpoint_url=... WHERE
session_id='worker-gpu-h100'`.

**Why:** the stored URL pointed at a dead tunnel, and the code's own recovery
path would have stopped a healthy, billing H100 and rebuilt it (~6 min). The user
approved the specific edit.

**Affects:** one field, one row. A safety copy was taken first
(`lab.db.bak-20260915-231512`, now gitignored). The underlying code bug is fixed
in `worker_manager.py` and documented as §92, so this shouldn't recur.
