# Deferred work

Items scoped or identified but **not** executed. Only genuinely open items live
here — when something is done, delete its entry (git history records how it was
resolved).

---

## 2026-09-16 — CPU trainer endpoint sits in ERROR state

`aiendpoint-e00aqam4wv0eydfkpc` (`llm-lab-cpu-trainer`) has been in `ERROR` since
at least 2026-09-15. The app handles this correctly — Decision 4 in
`worker_manager.ensure_worker` deletes and recreates an ERROR endpoint on next
use — so nothing is broken. But nothing will touch it until someone asks for a
CPU run.

**Status:** OPEN. Worth clearing before demoing the CPU path to anyone.

---

## Closed on 2026-09-16 (kept briefly for orientation, delete freely)

- **Gradient clipping** — done. nanoGPT now clips at 1.0 like Karpathy;
  config-driven so transformer/MoE are deliberately unchanged and older runs
  stay comparable. See `tests/test_grad_clip.py`.
- **Exported scripts ignored the inference config** — done. `GENERATE_TOKENS`
  and `TEMPERATURE` now come from config, and export prefers the *run's*
  snapshot over the experiment's current config.
- **top-k sampling** — decided against. User chose untruncated sampling, which
  shows the raw distribution and teaches more. Recorded in
  `docs/NANOGPT_VS_KARPATHY.md` §3.
- **`backend/export.py` over 500 lines** — threshold explicitly waived by the
  user; splitting it would be more confusing than the length.
