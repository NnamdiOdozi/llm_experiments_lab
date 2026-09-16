# nanoGPT: ours vs Karpathy's

**Verified 2026-09-16** by reading Karpathy's source directly, not from memory:

- `train.py` — https://raw.githubusercontent.com/karpathy/nanoGPT/master/train.py
- `model.py` — https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py
- `sample.py` — https://raw.githubusercontent.com/karpathy/nanoGPT/master/sample.py

Our config lives in `config/presets.py` (`NANOGPT_CONFIG`). Re-verify against the
links above before trusting this page, in case upstream has moved on.

---

## 1. The architecture is identical

Every architectural value matches his defaults exactly.

| Setting | Karpathy | Ours |
|---|---|---|
| `n_layer` | 12 | 12 |
| `n_head` | 12 | 12 |
| `n_embd` | 768 | 768 |
| `block_size` | 1024 | 1024 |
| `dropout` | 0.0 | 0.0 |
| `bias` | False *(see §4)* | False |
| `batch_size` | 12 | 12 |
| weight tying | yes | yes |
| weight-decay split | 2D decays, 1D doesn't | same |
| LR schedule | linear warmup → cosine | same |

So when someone asks "is this really nanoGPT?", the answer for the model itself
is yes.

## 2. What we changed, and why

All of our deviations are in the **training schedule**, because we run a short
demo on one GPU against Tiny Shakespeare rather than days on OpenWebText.

| Setting | Karpathy | Ours | Why |
|---|---|---|---|
| `dataset` | OpenWebText | Tiny Shakespeare | Small, fast, already in the lab |
| `max_iters` | 600,000 | 3,000 | Demo length, matches our tiny transformer |
| `learning_rate` | 6e-4 | 3e-4 | Halved; gentler for a short run |
| `min_lr` | 6e-5 | 3e-5 | Kept at LR/10, same ratio as his |
| `warmup_iters` | 2,000 | 200 | Scaled to our 3,000-step budget |
| `lr_decay_iters` | 600,000 | 3,000 | Cosine must finish within the run |
| `grad_accum_steps` | 40 (`5 * 8`) | 5 | His 40 assumes **8 GPUs**; 5 is the 1-GPU equivalent |
| `eval_interval` | 2,000 | 10 | So the loss curve moves while you watch |
| `eval_iters` | 200 | 2 | Same reason |
| `max_new_tokens` | 500 | 100 | Shorter sample for the UI |
| `temperature` | 0.8 | 0.8 | **Unchanged — this one is his** |

### The evaluation trade-off is worth understanding

We evaluate 200× more often on 1/100th of the data. Each validation point comes
from **2 batches**, so it is a noisy estimate. Our loss curve jitters in a way
his would not.

That is the right trade for a teaching tool — a curve that only updates every
2,000 steps shows nothing during a demo. But **do not read meaning into small
wiggles in our validation curve**. They are sampling noise, not learning.

## 3. Things Karpathy does that we do NOT

These are absent from our config entirely, so nothing in the UI reveals them.

### Gradient clipping — a real gap
He sets `grad_clip = 1.0` and applies it between backward and the optimizer step:

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

Our nanoGPT loop does **not** clip. `_train_char_lm` does `zero_grad`,
accumulates `backward()` over micro-batches, then `optimizer.step()` — nothing
in between. (Our **RNN** path does clip, at 5, so the codebase has the
capability; the nanoGPT path just doesn't use it.)

At 124M parameters with a 50k vocabulary, one unlucky batch can produce a very
large gradient. Clipping caps it. It is the standard guard against a run that
trains fine for hours then spikes into garbage.

**CLOSED 2026-09-16.** `_train_char_lm` now clips when `grad_clip` is set, and
`NANOGPT_CONFIG` sets it to 1.0 — Karpathy's value. The call is config-driven
(three lines, no template branching), so transformer and MoE deliberately stay
unclipped and runs from before this change remain comparable. Behaviour pinned by
`tests/test_grad_clip.py`, which asserts it fires once per step at 1.0 and *not*
at all when the key is absent or 0.0.

### Top-k sampling
He truncates generation to the 200 most likely tokens (`top_k = 200`). We don't
truncate. With 50k tokens the unlikely tail is large, so the model will
occasionally pick something absurd. This mattered far less on our 65-character
baseline. Arguably worth keeping untruncated for teaching — it shows the raw
distribution — so this is a judgement call, not a bug.

## 4. A genuine inconsistency inside his own repo

`bias` disagrees between his two files:

- `model.py` — `GPTConfig` defaults `bias = True`
- `train.py` — the training config sets `bias = False`

`train.py` wins in practice, so real nanoGPT training runs without bias. **We
use `False`, which matches how it actually trains.** Noted here because our
preset comment says `bias=False` while his model class appears to say otherwise.

## 5. Why the landing-page card says so little

The card previously advertised "warmup+cosine LR scheduler, weight-decay
optimizer, weight tying" under **Notable:**. All three are his defaults
(verified above), so calling them notable implied we had chosen them.

The card now names only what differs from our own tiny transformer baseline:
scale (124M vs 1.8M), vocabulary (real GPT-2 BPE vs character-level), and weight
tying — which the baseline genuinely does **not** do (it builds a plain
`nn.Linear` head, no sharing). The LR schedule and optimizer stay off the card:
they appear in the config pane as editable numbers, and they don't change what
the model *is*.

## See also

- `config/presets.py` — `NANOGPT_CONFIG`, plus the memory/OOM notes above it
- `backend/training/templates/nanogpt/model.py` — our model
- `backend/training/train_worker.py` — `_train_char_lm`, the shared loop
- `DEFERRED_WORK.md` — gradient clipping and top-k, still open
