# LLM Experiments Lab — Project Discussion Document

**Project stream:** Nebius Serverless AI Builders Challenge  
**Team:** Nnamdi, Denis Shipilov, + TBC  
**Date:** 24 June 2026  
**Status:** Ideation / Pre-development — incorporates external review feedback (see §6.2, §14)

---

## 1. The One-Liner

A browser-based laboratory where learners take concepts from an LLM theory course — attention, tokenization, MoE, positional encodings, hyperparameter tuning — and ground them in real experiments: swap components, retrain on real GPUs, watch the loss curve, and connect what they see to what they studied.

---

## 2. Why This Exists

There is a gap between understanding transformer theory in slides and *feeling* how the pieces interact. Jupyter notebooks fill that gap badly: they give unlimited freedom and zero structure, which is exactly wrong for someone grounding theory. The learner doesn't yet know which knobs matter.

The Lab's job is to deliberately remove degrees of freedom: expose the handful of levers that teach and hide the thousand lines that don't.

**The reference point:** [TensorFlow Playground](https://playground.tensorflow.org/) did this for simple neural networks. Nobody has done it well for transformers and modern architectures at a level that involves real training on real GPUs.

---

## 3. How the Lab Differs from a Notebook

| Capability | Jupyter Notebook | LLM Experiments Lab |
|---|---|---|
| Mid-training interaction | Training loop blocks the kernel; requires thread/callback hacks | Pause, prompt the half-trained model, resume — built in |
| Architecture visualisation | Read `nn.Module` code and imagine it | Live schematic that updates when config changes |
| Comparable results | Eyeball two bespoke matplotlib charts | Standardised loss readouts on identical axes |
| Contextual assistant | ChatGPT in another tab; no awareness of your run | Grounded chatbot that sees your config, loss curve, and last change |
| Experiment structure | Blank canvas, no expected outcomes | Lab manual with hypothesis → expected outcome |

**In short:** a notebook optimises for an expert who already knows which experiment to run. The Lab optimises for a learner who knows the theory but not yet which experiment reveals it.

---

## 4. The Challenge Context

The Nebius Serverless AI Builders Challenge is a community competition for developers, ML engineers, researchers, and students. The brief: build a serverless AI project on Nebius AI Cloud, document it clearly, share it openly.

**What we get:**
- $100 in Nebius compute credits for every valid submission
- Up to $2,000 in credits for top-3 placements
- Special award categories including "Student Projects" and "AI & ML"
- Winning submissions become reference examples in the Nebius docs/blog

**What Nebius Serverless AI provides (the three primitives):**
- **Jobs** — containerised finite workloads that start, run, and complete (our training runs)
- **Endpoints** — model serving behind HTTP endpoints (our inference/prompting endpoint)
- **DevPods** — interactive dev environments with Jupyter and VS Code (development)

Billing is per-second, pay-as-you-go. No cluster management, no GPU driver setup.

---

## 5. Architecture Streams

The project has two natural streams that share a UI but differ in what runs on the GPU.

### Stream 1: Transformer Lab (primary)

The baseline is the Tiny Transformer — a small decoder-only transformer trained from scratch (the course's `tiny_transformer_lm` notebook). Users can swap architectural components and retrain:

- **Tokenizer** — swap between BPE variants, character-level, etc.
- **Attention** — standard multi-head, grouped-query, multi-query
- **Positional encoding** — sinusoidal, RoPE, ALiBi, relative
- **Feed-forward block** — standard MLP vs. MoE (Mixture of Experts)
- **Hyperparameters** — learning rate, batch size, number of layers/heads, embedding dimension, dropout
- **Training data** — select from curated small datasets (Shakespeare, code, Wikipedia excerpts)

**The signature experiment:** Pause a run mid-training, prompt the half-trained model, watch the outputs sharpen as training resumes. "Loss going down" made tangible.

### Stream 2: Classical Architectures (extension)

**Status: Phase 2 — deferred from MVP.** An external review of this document (June 2026) recommended cutting Stream 2 from the challenge submission to sharpen positioning: "TensorFlow Playground, but for tiny transformers and LLM concepts" is a clearer, more demo-friendly wedge than a general-purpose ML lab. We agree — see §6.2.

Same Lab UI, different model family:

- **RNN / GRU / LSTM** — for time-series or sequence tasks. Compare gating mechanisms, observe vanishing gradient behaviour, contrast with transformer attention.

This stream broadens the Lab's audience beyond LLM learners and reinforces that the core concepts (loss landscapes, hyperparameter sensitivity, architecture choices) are universal — worth building once the Stream 1 MVP is solid.

---

## 6. Feature Breakdown

### 6.1 Must-Have (MVP for challenge submission)

| Feature | Description |
|---|---|
| **Config UI** | Browser interface to set neural network parameters and hyperparameters. Visual, not code-first. |
| **Live training on Nebius GPU** | Training runs dispatched as Nebius Serverless Jobs. Loss curve streams back in real time. |
| **Architecture schematic** | Visual diagram of the current model that updates live when the user changes config. The code and the abstraction are the same object. |
| **Pause-and-prompt** | Stop training mid-run, send a prompt to the half-trained model, see the output, resume training. |
| **Lab manual** | A set of prescribed experiments with hypothesis → expected outcome. "Enable MoE at fixed active params — predict and then measure the effect on loss." |
| **Grounded chatbot** | LLM assistant that sees the user's current config, loss curve, and last change. Can say: "Loss plateaued right when you enabled MoE — here's why." |
| **Public/private experiments** | Users can make their experiments public or private. Public experiments are viewable and runnable by anyone, but cannot be saved to another user's profile — to keep a copy, they must fork it (creating a new experiment under their own account). Private experiments are visible only to the owner. |
| **Templates** | A curated library of public starter configs that anyone can fork. The "general templates area" for the group. |
| **Code view + export** | Users can view the generated PyTorch code alongside the visual UI — see how the pieces are plugged together. Code is downloadable as `.ipynb` (Colab-ready) or `.py`. Editing in-browser is TBD: if users edit code directly it invalidates the no-code UI state, so the interaction model (edit via UI/chatbot only, or allow direct code edits that sync back) needs further discussion. |
| **Experiment notes** | A free-text notes panel attached to each experiment (not buried in a downloaded file) — see §6.1.1 for the reasoning. |

### 6.1.1 Where Experimenters Keep Notes

Notes live in the app, not in a downloaded file. A simple "Notes" tab sits alongside Config / Code / Results for each experiment — a plain markdown textarea, autosaved to the database as the user types (debounced, no explicit save button). This is deliberately separate from the auto-generated `README.md` mentioned in the Phase 2 export bundle (§6.2): the README is a structured, machine-written summary (hypothesis / fixed / changed / result); the Notes panel is the user's own freeform scratchpad — "tried X, didn't work because Y, want to try Z next."

Why in-app rather than a `.txt`/`.md` file: a downloaded file requires leaving the Lab to edit it and re-uploading or keeping it somewhere else entirely — exactly the kind of friction that causes notes to never get written. An in-app field is open in the same tab as the run, so jotting something down mid-experiment costs nothing.

Notes are stored at the **experiment level**, not per training run — so notes persist and accumulate across multiple runs of the same config (e.g., "run 1: loss too noisy, run 2: lowered LR, much better"). At export time (Phase 2, §6.2), the notes are included in the bundle as `notes.md`, separate from the auto-generated `README.md`, so nothing is lost when a user downloads their work.

### 6.2 Phase 2 — Post-MVP Backlog

This section was substantially expanded following an external review of this document (June 2026). Items below are ones the team agrees are good ideas, but are deliberately kept out of the MVP to avoid bloating a prototype build. Pure productionisation/scaling concerns (cost governance, dataset licensing, security hardening, etc.) live separately in §14 so none of the reviewer's work is lost, but they don't clutter this product backlog.

| Feature | Priority | Description |
|---|---|---|
| **Comparable instruments** | High | Standardised loss/metric readouts on identical axes so "my MoE run vs. baseline" is a real comparison, not eyeballing. |
| **Diff-from-baseline display** | High | When a user changes config from a template/baseline, show a clear diff (e.g., `n_layer: 4 → 8`, `learning_rate: unchanged`) rather than leaving them to spot differences themselves. Prevents the classic mistake of comparing a 4-layer RoPE model against a 6-layer sinusoidal model and drawing the wrong lesson. |
| **Evaluation metrics beyond loss** | High | Perplexity, parameter count, active parameter count (for MoE), tokens/sec, memory usage. Loss alone is hard for beginners to interpret — these make progress and trade-offs tangible. |
| **MoE-specific metrics** | High | Expert usage counts, load-balancing loss, tokens-per-expert. If MoE is a flagship experiment but expert routing is invisible, learners can't see *why* the loss moved — they just watch a number change with no visible mechanism. |
| **Expanded run status states** | Medium | `queued / starting / running / pause_requested / checkpointing / paused / resuming / completed / failed / cancelled / timeout` rather than just running/completed/failed. Cloud jobs fail, browsers close, users double-click buttons — the state machine should be explicit about it. |
| **Max run duration (job timeout)** | Medium | A per-job time limit so one stuck container doesn't quietly burn GPU-minutes indefinitely. |
| **Export and Portability bundle** | Medium | Beyond a single notebook download: a zip containing `config.json`, `train.py`, `model.py`, `data.py`, `generate.py`, `notebook.ipynb`, `metrics.jsonl`, `README.md`, `notes.md`, `requirements.txt`. The README auto-generates an experiment story: question asked, changed variable, fixed variables, result. `notes.md` carries over the user's own freeform notes (§6.1.1) unchanged. This gives a strong "documentation quality" story for future GitHub sharing without building GitHub push automation yet. |
| **Lightweight reproducibility metadata** | Medium | Store random seed and pinned package versions (`torch==2.x`, etc.) per run, so "rerunning the same experiment" actually reproduces the same result. (Heavier reproducibility tracking — Docker image tags, git commits — is a production concern; see §14.) |
| **Narrower chatbot modes** | Medium | Constrain the grounded chatbot to specific modes rather than open-ended Q&A: *explain this run*, *compare two runs*, *suggest next experiment*, *check my hypothesis*, *explain the code*. Keeps answers grounded in the actual run state rather than drifting into generic LLM chat. |
| **Formal experiment-mode taxonomy** | Low | Naming explicit modes (controlled comparison / bundled change / free exploration / template-guided) rather than relying on the diff-view alone. Worth revisiting if the diff-view proves insufficient in practice. |
| **MLflow integration** | Low | Track experiment runs, parameters, and metrics in MLflow for more sophisticated experiment management. |
| **TensorBoard integration** | Low | Embed TensorBoard visualisations (weight histograms, computation graphs) within the Lab UI. |
| **Stream 2 architecture** | Medium | RNN/GRU/LSTM model family with an appropriate sequence/time-series dataset (see §5). |
| **Collaborative features** | Low | Shared experiment sessions, commenting on public templates, forking with attribution. |
| **Leaderboard** | Low | Compare experiment results across users (e.g., lowest loss on a fixed dataset/param budget). |
| **Template versioning** | Low | Once templates are forked publicly, edits to a published template could break old forks or confuse old results. A `v1/v2/v3` model with immutable published versions avoids this — only matters once templates are forked at real scale. |

### 6.3 UI Interaction Model: Stackable Layer List

The Config UI uses a **stackable layer list** — a vertical list of layers representing the current architecture, with inline config panels. This approach sits between a free-form drag-and-drop canvas (too complex to build, too many invalid states) and a static sidebar of sliders (too rigid, doesn't let users modify architecture structure).

**How it works:**

1. The user sees the current architecture as a vertical stack: e.g., `Embedding → TransformerBlock × 4 → LayerNorm → Output`.
2. A "+" button between any two layers opens a layer picker: Dropout, BatchNorm, LayerNorm, Linear, etc.
3. The new layer snaps into the stack at that position.
4. The user clicks the layer to expand its inline config panel — a slider for dropout rate, a number input for hidden dimension, a toggle for bias, etc.
5. The architecture schematic (visual diagram) and code view both update immediately.

**Layer controls (on every layer in the stack):**

- **Delete (×)** — removes the layer entirely from the stack and the generated code.
- **Disable toggle** — greys the layer out without removing it. It stays visible in the stack (so the user remembers it was there) but is excluded from the generated code and training. Useful for A/B experiments: "run with BatchNorm enabled, disable it, run again, compare."
- **Repeat count (−/+)** — for blocks that repeat (e.g., `TransformerBlock × 4`), a stepper controls the count. Want 2 layers instead of 4? Click minus twice. Want 8? Click plus. This avoids making users manually add/delete identical blocks one at a time.
- **Reorder (drag)** — drag a layer up or down within the stack to change its position.

**Where different settings live:**

| Setting type | Where in the UI | Examples |
|---|---|---|
| Layer-level params | Inline panel on each layer in the stack | Dropout rate, number of attention heads, kernel size, hidden dim |
| Architecture-level choices | Top-level toggles/dropdowns above the stack | Positional encoding type (RoPE / sinusoidal / ALiBi), MoE vs. dense MLP |
| Training config | Separate "Training" panel | Learning rate, batch size, optimizer, L1/L2 regularisation weight, epochs |
| Data preprocessing | Separate "Data" panel | Dataset selection, standard scaler toggle, train/val split |
| Compute | Separate "Compute" panel | CPU (default) vs. GPU, device type |

**Why this model fits:**

The course homework notebooks are built as sequential layer stacks — the tiny transformer is `Embedding → [TransformerBlock × N] → LayerNorm → Linear`. A vertical list is the natural visual representation of architectures that are inherently sequential. It also maps cleanly to code generation: the layer list is literally a `nn.Sequential` or a `forward()` method that calls layers in order.

---

## 7. Technical Architecture (Draft)

```
┌──────────────────────────────────────────────────────────┐
│                     BROWSER (React)                      │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────────────┐ │
│  │  Config UI  │ │  Schematic   │ │  Loss Curve + Chat  │ │
│  │ (params,    │ │  (live arch  │ │  (streaming chart,  │ │
│  │  hyper-     │ │   diagram)   │ │   grounded chatbot) │ │
│  │  params)    │ │              │ │                     │ │
│  └─────┬──────┘ └──────────────┘ └──────────┬──────────┘ │
│        │                                     │           │
└────────┼─────────────────────────────────────┼───────────┘
         │           WebSocket / SSE           │
         ▼                                     ▼
┌──────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI)                            │
│                                                          │
│  • Receives config, dispatches Nebius Serverless Job     │
│  • Polls Object Storage for metrics, relays to browser  │
│  • Handles pause/resume signals                         │
│  • Proxies chatbot queries to Nebius Token Factory       │
│  • Manages templates and experiment storage              │
│  • Enforces max 3 concurrent runs per user               │
└────────┬──────────────────┬──────────────────┬───────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ NEBIUS        │  │ NEBIUS OBJECT    │  │ NEBIUS TOKEN     │
│ SERVERLESS    │  │ STORAGE (S3)     │  │ FACTORY          │
│               │  │                  │  │                  │
│ Jobs: train   │  │ metrics.jsonl    │  │ LLM inference    │
│ runs on       │→ │ checkpoint.pt    │  │ for chatbot      │
│ GPU/CPU       │  │ model outputs    │  │ (Qwen3-235B)     │
│               │  │                  │  │                  │
│ Endpoints:    │  │ Permanent record │  │ Embeddings for   │
│ pause-and-    │  │ of all runs      │  │ template search  │
│ prompt        │  │                  │  │                  │
└──────────────┘  └──────────────────┘  └─────────────────┘
```

**Key mapping to Nebius primitives:**

| Nebius Service | Our Use |
|---|---|
| Serverless Jobs | Each training run is a Job. Container starts, trains for N steps, writes metrics + checkpoints to Object Storage, exits. |
| Serverless Endpoints | Serve the half-trained model checkpoint for pause-and-prompt inference. Spun up on demand, torn down after. |
| Object Storage (S3) | Stores `metrics.jsonl` (the live loss data), `checkpoint.pt` (model weights), and model outputs. This is both the real-time streaming mechanism and the permanent record. |
| Token Factory | Powers the grounded chatbot (LLM inference) and template search (embeddings). Cheap, always-on, OpenAI-compatible API. |

### 7.1 Metrics Pipeline: How Loss Data Gets to the Browser

No Prometheus, no Langfuse, no log aggregator. The pipeline is three components passing a JSONL file:

**Component 1 — Training container (Nebius Serverless Job):**

The PyTorch training loop has ~3 extra lines. Every N steps (e.g., every 10), it appends a JSON line to a local file and uploads it to Object Storage:

```python
for step in range(num_steps):
    loss = train_step(model, batch)

    if step % 10 == 0:
        row = {"step": step, "train_loss": loss.item(), "val_loss": val_loss}
        with open("/tmp/metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        s3.upload_file("/tmp/metrics.jsonl", bucket, f"runs/{run_id}/metrics.jsonl")
```

The file grows by one line every 10 steps. Each upload overwrites the previous version. No SDK, no agent, no daemon — just a file write and an S3 PUT.

**Component 2 — CPU backend (FastAPI):**

A background task polls the metrics file from Object Storage every 2-3 seconds and pushes only the new lines to the browser via WebSocket:

```python
async def poll_metrics(run_id: str, websocket: WebSocket):
    last_line_count = 0
    while run_is_active(run_id):
        obj = s3.get_object(bucket, f"runs/{run_id}/metrics.jsonl")
        lines = obj.read().decode().strip().split("\n")
        for line in lines[last_line_count:]:       # only NEW lines
            await websocket.send_text(line)
        last_line_count = len(lines)
        await asyncio.sleep(2)
```

The backend is a relay — it doesn't process or aggregate anything. Read from S3, push to WebSocket, repeat.

**Component 3 — Browser (React):**

A WebSocket listener appends each incoming point to the chart data:

```javascript
ws.onmessage = (event) => {
    const point = JSON.parse(event.data);
    setLossData(prev => [...prev, point]);
    // Recharts re-renders automatically
};
```

**Why this works:** a training step on the tiny transformer takes ~50-200ms, so "every 10 steps" produces a new data point every 0.5-2 seconds. The 2-second polling interval matches the data production rate. Nobody will perceive the difference between this and true real-time. And the metrics file in Object Storage serves double duty: it's the live streaming source during the run, and the permanent record after the run. When a user revisits a past experiment, the backend reads the same file and renders the full loss curve — no separate logging system needed.

**Checkpoints follow the same pattern:** the container saves `checkpoint.pt` to Object Storage every N steps (e.g., every 500). When the user clicks "Pause," the backend signals the container (via a flag file in Object Storage or by terminating the Job), the container saves a final checkpoint and exits gracefully.

---

## 8. The Lab Manual — Example Experiments

These are the prescribed experiments that ship with the Lab. Each follows the pattern: **context → hypothesis → experiment → expected outcome → what to look for.** Ordered to build up gradually: Tiny Transformer fundamentals first, then LoRA fine-tuning, then MoE last — each one builds on intuition from the previous.

1. **Baseline run** — Train the default Tiny Transformer baseline config for 5000 steps. Establish the baseline loss curve. Understand what "normal" looks like.

2. **The signature experiment: pause-and-prompt** — At step 500, 1500, 3000, and 5000, pause and prompt the model with "The king said to the queen." Watch outputs go from random tokens to coherent (if odd) English. *Loss going down, made tangible.*

3. **RoPE vs. sinusoidal positional encoding** — Swap positional encodings, keep everything else identical. *Hypothesis: RoPE should show better extrapolation on longer sequences. Measure: loss on sequences 2× the training length.*

4. **Learning rate sensitivity** — Run the same config at 3e-4, 1e-3, and 3e-3. *Hypothesis: too-high LR will cause loss spikes. Watch for the classic "loss spike then recovery" pattern.*

5. **Tokenizer swap** — Train with character-level tokenizer vs. BPE-1024 vs. BPE-8192. *Hypothesis: BPE reduces sequence length and speeds convergence, but character-level gives finer-grained attention patterns.*

6. **LoRA fine-tuning** — Freeze a pretrained GPT-2 124M and attach low-rank adapters instead of full fine-tuning. *Hypothesis: a small number of trainable parameters (the LoRA adapters) can adapt the frozen model nearly as well as full fine-tuning. Measure: loss vs. trainable parameter count, compared against a full-fine-tune baseline.*

7. **MoE at fixed active params** — Replace the dense MLP with a 4-expert MoE layer, keeping active parameters constant. *Hypothesis: loss should decrease because specialisation helps. Measure: loss delta at step 3000, and (Phase 2) expert usage distribution — see §6.2.*

8. **(Stream 2 — Phase 2) LSTM vs. GRU vs. Transformer on time-series** — On a synthetic sine-wave prediction task, compare all three. *Hypothesis: Transformer will overfit on small data; LSTM/GRU will do better with less data. Observe the crossover point as dataset size increases.*

---

## 9. The Grounded Chatbot

The chatbot is not a generic LLM. It is a **lab demonstrator** — grounded in the user's current experiment state.

**What it sees:**
- Current model config (architecture, hyperparameters)
- Live loss curve (current step, loss value, trend)
- Last change the user made ("you just enabled MoE")
- The lab manual entry for the current experiment (expected outcome)

**What it does:**
- Connects what the user is seeing to what they studied: "That loss plateau is consistent with the MoE routing collapse described in the Switch Transformer paper."
- Troubleshoots: "Your loss spiked after increasing LR to 3e-3. Try cosine annealing with warmup."
- Prompts reflection: "You predicted MoE would lower loss by 10%. It actually lowered it by 3%. Why might that be?"

**Powered by:** Nebius Token Factory (e.g., Qwen3-235B-A22B-Thinking or similar reasoning model). The experiment context is injected into the system prompt.

---

## 10. Templates and Social Features

- **General templates** are public, curated starter configs maintained by the team. Think of them as the "lab stations" — pre-configured experiments that are ready to fork and run.
- Users can **start from scratch** or **fork a template**. Forking copies the config and lets the user modify it.
- Each experiment has three views: **visual UI** (the config panel + schematic), **code view** (the generated PyTorch code), and **results view** (loss curves, metrics, model outputs).
- Experiments can be toggled **public** or **private**. Public experiments are discoverable and forkable by others.

---

## 11. Infrastructure: Auth, Database, Sessions

### Authentication

GitHub OAuth + Google OAuth (both free). GitHub covers developers and students; Google covers everyone else (factory workers, non-technical learners, anyone with a Gmail). Both use the same flow: user clicks "Sign in with GitHub/Google" → provider redirect → callback with token → backend verifies and issues a signed JWT cookie. Libraries: `authlib` or `httpx-oauth` for FastAPI — both providers are supported out of the box.

No passwords, no email verification, no user management. The OAuth providers handle all of that.

### Database

PostgreSQL (managed, or SQLite for early prototyping). Five core tables:

```
users
  id, auth_provider (github/google), provider_id,
  username, email, avatar_url, created_at

experiments
  id, user_id, config_json, is_public, forked_from_template_id,
  notes_md, created_at, updated_at

training_runs
  id, experiment_id, status (queued/running/paused/completed/failed),
  device (cpu/gpu), nebius_job_id,
  train_loss_history (JSON array), val_loss_history (JSON array),
  final_train_loss, final_val_loss,
  total_steps, current_step,
  checkpoint_url, metrics_url,
  started_at, completed_at

templates
  id, title, description, config_json, lab_manual_entry,
  fork_count, created_by (nullable — null = team-curated)

forks
  id, user_id, template_id, experiment_id, created_at
```

An **experiment** is a configuration (architecture + hyperparameters + dataset). A **training run** is a single execution of that configuration. One experiment can have many runs — the user tweaks nothing and re-runs to check variance, or pauses and resumes (which creates a new run from the checkpoint). When a user logs in and views their history, they see experiments, and under each experiment, the runs with their loss curves and status.

`config_json` stores the full experiment configuration as a JSON blob — architecture choices, hyperparameters, dataset selection, layer stack, everything. This is the single source of truth for reproducing any experiment.

### Training Run History

When a user logs in, their dashboard shows:

- A list of their experiments (each with its config summary: "Transformer, 4 layers, RoPE, lr=3e-4")
- Under each experiment, the training runs with status, train/val loss, duration, and device used
- Click any run to see the full loss curve, the config that produced it, and the model outputs from any pause-and-prompt interactions
- Side-by-side comparison: select two or more runs and overlay their loss curves on the same axes

All of this is just database queries against the `training_runs` table. The loss history is stored as a JSON array of `{step, train_loss, val_loss}` objects — small enough to store inline for the run lengths we're dealing with (a few thousand steps).

### Parallel Training (Multi-User)

Each training run is an independent Nebius Serverless Job. Ten users clicking "Run" at the same time means ten separate Jobs, each provisioned with their own compute by Nebius. They don't queue behind each other or share resources — this is true parallelism, not time-sliced concurrency.

The backend (FastAPI with `async`) handles concurrent API requests from multiple users natively. Each user is limited to **3 concurrent training runs** to prevent accidental cost blowout. When a user clicks "Run," the backend:

1. Checks the user's active run count — if already at 3, returns an error ("You have 3 runs in progress. Wait for one to finish or cancel it.")
2. Writes a `training_runs` record with status `queued`
3. Calls the Nebius Serverless Jobs API to create a Job (HTTP POST — non-blocking)
4. Updates status to `running` when Nebius confirms the Job started
5. Polls Object Storage for metrics updates (see §7.1), relays to browser, writes to database
6. Updates status to `completed` or `failed` when the Job finishes

This scales naturally: more users = more Nebius Jobs = more parallel GPUs. The backend itself stays lightweight because it's just dispatching work and storing results, not running any training.

**Cost implication:** parallel usage means parallel billing. If 20 users each run a 5-minute GPU Job simultaneously, that's 100 GPU-minutes billed. The $100 in Nebius credits from the challenge submission helps here, and CPU-default mode keeps costs low for beginners.

### Session and Activity Tracking

The JWT cookie *is* the session — no server-side session store. Activity tracking is just database writes: creating an experiment writes to `experiments`, forking writes to `forks`. "Which notebooks did this user work on" is `SELECT * FROM experiments WHERE user_id = ?`.

Public/private visibility is a boolean column on `experiments`. Templates are always public.

### Python Dependencies

Two separate dependency sets:

**Web app (CPU backend):**
- `fastapi`, `uvicorn` — API server
- `authlib` or `httpx-oauth` — GitHub + Google OAuth
- `asyncpg` — PostgreSQL client (or `aiosqlite` for SQLite)
- `boto3` or `aiobotocore` — S3 client for Nebius Object Storage
- `nbformat` — programmatic notebook generation for export/download
- `httpx` — calling Nebius Serverless APIs

**Training containers (one per architecture stream):**
- Stream 1 (Transformer): `torch`, `numpy`, `matplotlib`
- Stream 1 (LoRA): above + `transformers`, `datasets`
- Stream 2 (RNN, Phase 2): `torch`, `numpy`, `matplotlib`

Each training container is a separate Docker image. The web app never imports PyTorch.

---

## 12. Source Notebooks (Course Homework)

These notebooks from the Nebius Academy course are the raw material the Lab is built from. Each one becomes a template or experiment type.

| Notebook | Cells | Architecture | Dataset | Device | Lab Role |
|---|---|---|---|---|---|
| `tiny_transformer_lm` | 41 (23 code, 18 markdown) | Decoder-only Transformer from scratch | Tiny Shakespeare (char-level) | CPU-friendly | Stream 1 baseline — the starting point |
| `homework_lora` | 50 (29 code, 21 markdown) | LoRA from scratch on frozen GPT-2 124M | TinyShakespeare via HF | GPU required | Stream 1 experiment 2 — fine-tuning |
| `4_2_tiny_moe_lm_hometask` | 35 (20 code, 15 markdown) | Extends the baseline with RoPE + MoE layers | Tiny Shakespeare | CPU/GPU | Stream 1 experiment 3 — component swap, last in sequence |
| `RNN_LM_homework` | 55 (28 code, 27 markdown) | RNN / LSTM / GRU language model | External dataset (gdown) | CPU-friendly | Stream 2 (Phase 2) — classical sequence modelling |

Note: a CIFAR-10/CNN notebook was attached in chat at one point but was never part of the actual course material or project sources, and has been removed from scope accordingly — no CNN stream is planned.

---

## 13. Open Questions

| Question | Notes |
|---|---|
| **Baseline model** | ~~Resolved~~ — the baseline is the course's own Tiny Transformer notebook (`tiny_transformer_lm`), not a fork of Karpathy's nanoGPT. They are different codebases; no need to introduce nanoGPT into the project. |
| **How does pause-and-prompt work mechanically?** | The training Job needs to checkpoint and exit. Then we spin up a Serverless Endpoint to serve inference on that checkpoint. Then we start a new Job from the checkpoint to resume. Is the cold-start latency acceptable? |
| **State persistence** | Where do experiment configs, checkpoints, and results live? Nebius Object Storage is the natural answer, but we need to design the folder/key structure. |
| **Frontend hosting** | The React UI needs a home. Vercel/Netlify for the frontend, with the FastAPI backend as a Nebius Endpoint? Or everything in one container? |
| **Scope for the challenge** | The challenge rewards "build something real, document it clearly." A focused MVP (Stream 1 only, 3-4 experiments, chatbot, templates) is better than a sprawling half-finished platform. |
| **Team roles** | Who owns frontend, backend/infra, ML/training code, lab manual content? |
| **MLflow / TensorBoard** | Are these worth the integration complexity for v1, or do we build a simpler bespoke metrics dashboard? Leaning toward bespoke for MVP. |
| **Cost envelope** | Each training run on a single GPU for ~5 minutes might cost a few cents. But if many users run many experiments, it adds up. Do we need usage caps or queuing? |

---

## 14. Productionisation — Out of Scope for Challenge MVP

This section exists so that none of the external reviewer's thinking gets lost, even though these items are explicitly **not** part of the prototype build. They matter once this moves from "challenge submission used by a known cohort" to "public product used by strangers on the internet." Revisit this section before any wider launch.

### Cost and Resource Governance

| Control | Why it matters at scale |
|---|---|
| Per-user daily GPU-minute quota | Stops runaway usage beyond the 3-concurrent-run cap already in the MVP (§11). |
| Max checkpoint size / storage cap | Prevents Object Storage costs growing unbounded. |
| Auto-delete old checkpoints | Keeps storage costs predictable over time. |
| Admin kill switch | Essential once the system is unattended — e.g., during a demo or public judging period, you want a way to halt all running Jobs instantly. |
| Queue instead of reject | Better UX than "try again later" once concurrent demand regularly exceeds capacity. |

### Heavier Reproducibility Tracking

Beyond the lightweight seed + package-version tracking already in the Phase 2 backlog (§6.2): Docker image tag, exact git commit of the training/app code, and a versioned config schema (`config_schema_version`). Matters once experiments need to be reproducible months later by people outside the original team — not needed for a challenge demo running on code that doesn't change underneath users.

### Dataset Governance

Redistribution rights (can a public template legally bundle this dataset?), content moderation (generated text from public datasets may surprise learners), determinism of train/val splits, and caching strategy to avoid repeated downloads. For the MVP, two or three curated, known-safe datasets sidestep all of this.

### Security and Sandboxing for User-Edited Code

If in-browser code editing (the open question in §6.1 — "Code view + export") is ever enabled, arbitrary Python execution on shared compute needs a formal security model: validated config fields only, never raw code execution, training containers that only run approved templates. This becomes critical *if* public experiments are forkable and editable by strangers. The MVP avoids the problem entirely by keeping code view-only.

### Database Scaling

Once the team needs to query across hundreds of runs ("show me all runs where validation loss dropped below 2.0," "show me all MoE runs with expert collapse"), a normalized `run_metrics` table (one row per step, indexed by run) becomes worth the complexity. For the MVP's scale, the JSON-blob-in-`training_runs` approach (§11) is simpler and entirely adequate — Object Storage remains the source of truth either way.

### GitHub Push Automation

The Phase 2 export bundle (§6.2) gives users a zip they can manually push to GitHub. Automating that — direct repo push, pull requests, Gist export — adds real complexity: OAuth write-scope consent (more sensitive than read-only login), repository selection UI, safe commit/overwrite behaviour, and credential handling (the app must never export tokens or secrets). Worth real product investment later; not needed to prove the concept.

### Public Gallery Attribution at Scale

If the public template/fork model (§6 and §10) grows into a real community gallery, each public experiment should eventually show: original author, fork lineage, config version, run status (don't surface broken runs), dataset used, and hardware used (for fair comparison). Template versioning (already noted as Phase 2 in §6.2) becomes more important here too.

### Reviewed and Intentionally Not Adopted

One reviewer suggestion we considered and are **not** adopting: replacing the overwrite-the-whole-file metrics approach (§7.1) with numbered per-chunk objects (`runs/{run_id}/metrics/000010.json`, `000020.json`, ...). The underlying technical point is correct — S3-compatible storage doesn't truly support append, so each upload rewrites the whole object. But at MVP scale (a few thousand training steps, a metrics file of a few hundred KB), re-uploading the whole file every 10 steps is trivial. Numbered chunks would require the backend to track the highest chunk seen, handle gaps, and garbage-collect old chunks — more moving parts for no practical benefit at this size. Worth revisiting only if run lengths grow by orders of magnitude.

---

## 15. Competitive Landscape and Positioning

| Tool | What it does | How we're different |
|---|---|---|
| TensorFlow Playground | Interactive neural net visualisation in the browser | No real training, no transformers, no GPU, no LLM components |
| Google Colab / Jupyter | General-purpose notebook with GPU runtime | No structure, no lab manual, no grounded assistant, no architecture visualisation |
| Weights & Biases / MLflow | Experiment tracking and comparison | Tracking tools, not teaching tools. No prescribed experiments, no pause-and-prompt |
| Andrej Karpathy's nanoGPT | Minimal GPT training code | Code-only, no UI, no guided experiments, expert-oriented |
| Hugging Face Spaces | Host ML demos | Demo hosting, not an experiment lab with training on real GPUs |

**Our unique position:** The only tool that combines real GPU training, guided experiments with expected outcomes, mid-training model interaction, architecture visualisation, and a grounded LLM assistant — purpose-built for learners bridging theory to practice.

---

## 16. Immediate Next Steps

1. **Agree on MVP scope** — Stream 1 only? How many lab manual experiments? Chatbot in v1 or v2?
2. **Spike: pause-and-prompt on Nebius** — Can we checkpoint a Tiny Transformer training Job, serve inference on the checkpoint via a Serverless Endpoint, and resume training from the checkpoint? Measure the cold-start latency. This is the riskiest technical question.
3. **Spike: streaming metrics** — Can a Nebius Job stream loss values back to a browser in near-real-time? (Probably via writing to object storage + polling, or via a lightweight WebSocket relay.)
4. **Design the config schema** — What parameters are exposed to the user? This defines the UI, the code generation, and the lab manual.
5. **Assign roles and set up the repo.**

---

*This document is a living draft. It captures the current state of thinking as of 23 June 2026 and will evolve as the team aligns on scope and begins development.*
