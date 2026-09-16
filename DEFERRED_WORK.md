# Deferred work

Items that were scoped or identified but **not** executed. Only genuinely open
items live here — when something is done, delete its entry (git history is the
record of how it was resolved).

---

## 2026-09-16 — nanoGPT has no gradient clipping

**Source:** verifying our nanoGPT config against Karpathy's `train.py`.

Karpathy sets `grad_clip = 1.0` and applies `clip_grad_norm_` between the
backward pass and the optimizer step. Our `_train_char_lm` in
`backend/training/train_worker.py` does `zero_grad` → accumulate `backward()` →
`optimizer.step()` with nothing in between. So nanoGPT (and transformer, and
MoE) train unclipped.

Our RNN path *does* clip, at 5 (`train_cfg.get("clip", 5)`), so the capability
exists and just isn't used here.

**Why it matters:** at 124M parameters with a 50k vocabulary, a single unlucky
batch can produce a very large gradient. Clipping is the standard guard against
a run that trains cleanly for hours then spikes into garbage.

**Scope:** one `nn.utils.clip_grad_norm_` call in `_train_char_lm`, plus a
`grad_clip` key in the presets (no hardcoding). Small. Needs a decision on
whether it applies to all char-LM templates or nanoGPT only.

**Status:** OPEN. Raised with the user 2026-09-16; not actioned.

---

## 2026-09-16 — exported scripts ignore the inference config

**Source:** found while adding nanoGPT/MoE export builders.

The generated standalone script hardcodes `GENERATE_TOKENS = 500` and never uses
`temperature` at all — its `generate()` samples straight from the softmax. Both
values sit in every preset's `inference` block (nanoGPT: `max_new_tokens` 100,
`temperature` 0.8) and are simply not threaded through.

This is **pre-existing**, not introduced by the nanoGPT/MoE work — it affects the
transformer export too.

**Why it matters:** the exported script is sold as "reproduce this experiment".
It currently reproduces the training but not the sampling settings. It also
violates the no-hardcoding rule.

**Scope:** thread two config values into the hyperparameter block and into
`generate()`. Small, but it changes generated output, so the transformer golden
in `tests/fixtures/export/` must be regenerated deliberately
(`tests/regen_export_goldens.py`) and the diff reviewed.

**Status:** OPEN.

---

## 2026-09-16 — top-k sampling absent from generation

**Source:** same comparison as above.

Karpathy's `sample.py` uses `top_k = 200`. We don't truncate at all. With a 50k
vocabulary the unlikely tail is large, so generation will occasionally pick
something absurd — much less of an issue on the 65-character baseline.

**Judgement call, not a defect:** untruncated sampling arguably teaches better,
since it shows the raw distribution. Needs a product decision before any code.

**Status:** OPEN, awaiting a decision rather than implementation.

---

## 2026-09-16 — backend/export.py is 695 lines

Past the 500-line threshold in CLAUDE.md. It now holds: tokenizer-code helpers,
module-level script fragments (`_MOE_CLASSES`, `_ADAMW_*`, `_LOOP_*`), the shared
`_char_lm_script`, `_rnn_script`, the dispatcher, and notebook/bundle building.

**Suggested split:** move the script *fragments* (the big template strings) into
`backend/export_fragments.py`, leaving `export.py` as assembly + dispatch. The
golden snapshots make this safe to do — any accidental output change fails.

**Status:** OPEN. Not urgent; flagged so it doesn't drift further.

---

## 2026-09-16 — CPU trainer endpoint sits in ERROR state

`aiendpoint-e00aqam4wv0eydfkpc` (`llm-lab-cpu-trainer`) has been in `ERROR` since
at least 2026-09-15. The app handles this correctly (Decision 4 deletes and
recreates on next use), so nothing is broken — but nothing will touch it until
someone requests a CPU run.

**Status:** OPEN. Worth clearing before demoing the CPU path.
