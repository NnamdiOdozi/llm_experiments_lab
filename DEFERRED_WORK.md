# Deferred work

Items scoped or identified but **not** executed. Only genuinely open items live
here — when something is done, delete its entry (git history records how it was
resolved).

---

## 2026-09-16 — nanoGPT overfits Tiny Shakespeare badly; corpus needs rethinking

**Source:** user observation — 124M params is a very large model for a 1.1 MB corpus.

**Confirmed, with numbers.** Tiny Shakespeare is 1,115,394 chars = **338,025
GPT-2 BPE tokens**. The nanoGPT preset trains on 12 x 5 x 1024 = 61,440 tokens
per step for 3,000 steps = **184.3M tokens**, i.e. **545 epochs** over the same
1.1 MB. Chinchilla-optimal for 124M params is roughly 2.5B tokens, so we are
about **7,300x short** on unique data. Roughly 367 parameters per unique token.

**Already visible in run 286** (nanoGPT, paused at step 431):

| step | train | val | val - train |
|-----:|------:|----:|------------:|
| 110 | 5.0445 | 5.3703 | +0.33 |
| 210 | 3.9600 | **4.7461** (best val) | +0.79 |
| 310 | 3.0594 | 5.1472 | +2.09 |
| 430 | 1.3853 | 5.6818 | +4.30 |

Validation loss bottoms out around **step 210** — 7% into the run — then climbs
steadily while training loss keeps falling. The remaining ~93% of the run is
pure memorisation. Any "nanoGPT reaches 124M params" demo is currently also a
demo of severe overfitting, whether or not that is the intended lesson.

**The useful target is not Chinchilla.** We do not need 2.5B tokens. We need
enough unique text that the existing 3,000-step budget stops being 545 passes.
**~184M tokens makes it a single pass.** Anything in the 100-200M range gets us
to roughly 1-2 epochs, which is a healthy regime.

**Candidate corpora (token counts approximate):**

| Corpus | ~tokens | Notes |
|---|---:|---|
| WikiText-103 | ~103M | Modern encyclopedic English, clean, standard benchmark |
| TinyStories | ~470M | Simple modern English, synthetic, very easy to learn |
| OpenWebText | ~9B | What Karpathy actually uses. ~54 GB — too big for this lab |
| FineWeb-Edu sample | configurable | Can take a ~200M-token slice |

**The real tension, and why this needs a decision rather than a default.**
Changing the corpus **breaks comparability with the tiny transformer baseline**.
Every existing run trained on Tiny Shakespeare; the landing page compares 124M
against 1.8M params on the same data. Swapping the corpus for nanoGPT only means
the two presets are no longer measuring the same thing. Options:

1. **Keep Shakespeare, add a modern corpus on top** (user's preference). Keeps
   continuity; mixing ratios become a new thing to explain and tune.
2. **Modern English only for nanoGPT.** Cleanest ML story, worst comparability.
3. **Keep Shakespeare, cut `max_iters` to ~250.** Zero new data, stops just past
   the val-loss minimum. Cheapest honest fix, but the model stays undertrained
   and a 3,000-step demo becomes a 250-step one.
4. **Keep it as-is and teach the overfitting.** The divergence above is a
   genuinely good teaching artefact — arguably the clearest one in the lab.

**Scope:** option 3 is a one-line config change. Options 1 and 2 need a data
loader, a download/caching path, `.gitignore` hygiene for a large corpus, and a
decision on tokenizer stats. Not small.

**Status:** OPEN. Needs a product decision first, not code.

---
