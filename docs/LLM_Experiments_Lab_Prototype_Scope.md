# LLM Experiments Lab — Prototype Scope (for Coding Agent Handoff)

**Purpose:** Define three buildable tiers of the prototype, each a strict superset of the previous one, so the coding agent can build incrementally and you can demo at any stage.

**Companion document:** `LLM_Experiments_Lab_Project_Discussion.md` — this file is a build-scoped subset of that broader product document. Where they conflict, this file wins for prototype purposes; the other file remains the source of truth for product direction.

---

## A Scope Simplification — Please Confirm

To keep the prototype demoable rather than a half-built platform, this spec **drops the following from all three tiers**, even though they're in the main project document:

- User authentication (GitHub/Google OAuth) — the demo runs as a single implicit user, no login screen.
- Multi-user public/private experiments, forking, and template social features — replaced with 2-3 hardcoded preset configs you can load.
- Cost governance, quotas, kill switches — not needed for a demo you control.

These all remain real future work (see `LLM_Experiments_Lab_Project_Discussion.md` §6.2 and §14) — they're just not part of what gets demoed at any of the three tiers below. **If you actually need login or multi-user behaviour in the demo itself, flag it now** — it changes the Tier 1 backend shape (you'd need at least a stub session) and adds real time to every tier.

---

## 1. Feature Scope by Tier

| Capability | **Tier 1**<br>Local machine, no Docker, no chatbot | **Tier 2**<br>Nebius Serverless + Docker | **Tier 3**<br>Nebius Serverless + Docker + Chatbot |
|---|---|---|---|
| **Frontend** | React (Vite), runs on `localhost` | Same | Same |
| **Config UI (layer stack)** | ✅ Full — add/remove/disable/reorder/repeat-count, on ~3 swappable knobs (positional encoding, dropout, layer count) | ✅ Unchanged | ✅ Unchanged |
| **Architecture schematic** | ✅ Yes, updates live with config | ✅ Unchanged | ✅ Unchanged |
| **Code view + export** | ✅ View generated PyTorch code; download `.ipynb`/`.py` | ✅ Unchanged | ✅ Unchanged |
| **Lab manual experiments** | 2-3 hardcoded (baseline run, RoPE vs. sinusoidal) | Same set | Same set, optionally + LoRA if time allows |
| **Training execution** | Local subprocess: training runs in a separate Python process launched by FastAPI, communicating via file-based IPC (status.json, metrics.jsonl, flag files) | Out-of-process: dispatched as a **Nebius Serverless Job** running inside a **Docker container** | Unchanged from Tier 2 |
| **Containerisation** | ❌ None — training runs as local subprocess, not containerised | ✅ Dockerfile for the training image; build + push to a registry | ✅ Unchanged |
| **Metrics streaming** | Local file (`metrics.jsonl` on disk) polled by backend, pushed to browser via WebSocket | Same pattern, but the file lives in **Nebius Object Storage (S3-compatible)** instead of local disk | ✅ Unchanged |
| **Pause-and-prompt** | Checkpoint-based: worker saves checkpoint and exits on pause; API loads checkpoint temporarily for inference; resume launches new worker from checkpoint | Real implementation: checkpoint → Job exits → spin up a **Nebius Serverless Endpoint** on that checkpoint → inference → tear down → new Job resumes | ✅ Unchanged |
| **Checkpointing** | Local `.pt` file on disk | Uploaded to Nebius Object Storage | ✅ Unchanged |
| **Database** | SQLite (single file, no server) | SQLite is fine to keep — no need to upgrade to Postgres just for a demo | ✅ Unchanged |
| **Grounded chatbot** | ❌ None | ❌ None | ✅ Calls **Nebius Token Factory**; sees current config + loss history + last change. 2-3 modes (e.g. "explain this run") is enough |
| **Auth / multi-user** | ❌ Out of scope — see note above | ❌ Out of scope | ❌ Out of scope |
| **Public/private/templates/forking** | ❌ Out of scope (2-3 hardcoded presets only) | ❌ Out of scope | ❌ Out of scope |
| **Deployment** | Runs on your laptop, nothing to deploy | Backend can still run on your laptop — only the *training* leaves your machine | Same |

---

## 2. Estimated Lines of Code

These are **rough order-of-magnitude estimates for newly-written application code** — they exclude: framework boilerplate (Vite/CRA scaffolding, `package.json`, lockfiles), the existing PyTorch model/training logic already written in your `tiny_transformer_lm` notebook (counted once, lightly, since it's adapted not authored from scratch), and tests (per your preference for a lean PoC with minimal error handling).

| Code Area | Tier 1 (cumulative) | Tier 2 (cumulative) | Tier 3 (cumulative) |
|---|---|---|---|
| Frontend (React) | 700–1,000 | 750–1,100 | 900–1,300 |
| Backend API + orchestration (FastAPI) | 350–500 | 600–850 | 750–1,050 |
| Training/model code (adapted from notebook + pause hook) | 150–250 | 170–290 | 170–290 |
| Code generator (config → PyTorch code templating) | 100–150 | 100–150 | 100–150 |
| Docker/infra (Dockerfile, build scripts) | 0 | 60–100 | 60–100 |
| Chatbot (Token Factory client, context builder, chat UI) | 0 | 0 | 200–280 |
| **Total** | **≈1,300–1,900** | **≈1,700–2,500** | **≈2,200–3,200** |

**Reading this table:** the jump from Tier 1 → Tier 2 is not "add a deployment step on top of Tier 1" — a meaningful chunk of Tier 1's training-orchestration and pause-and-prompt code gets **replaced**, not extended (see Gotcha #2 below). The jump from Tier 2 → Tier 3 is comparatively cheap and additive.

---

## 3. Gotchas for the Coding Agent

1. **Nebius's API is gRPC/Protobuf-based, not plain REST/JSON.** There's an official Python SDK (`pip install nebius`) that wraps this and handles auth-token renewal automatically — use it rather than hand-rolling raw gRPC calls. Budget time to learn its request-object pattern; it's not a `requests.post(...)` one-liner.
2. **Pause-and-prompt architecture.** Tier 1 now uses checkpoint-based pause: worker saves checkpoint and exits, API loads checkpoint for inference, resume launches a new worker. This is structurally similar to Tiers 2/3 (checkpoint → exit → inference → resume), making the migration path smoother. The main difference in Tiers 2/3 is that the worker is a Nebius Job and inference uses a Serverless Endpoint instead of loading in the API process.
3. **Object Storage is genuinely S3-compatible** — plain `boto3` with a custom `endpoint_url` works, confirmed against Nebius's own docs. No surprises here, this part of the Tier 2 estimate is solid.
4. **Cold-start latency for Jobs/Endpoints is unmeasured.** Before building out the rest of Tier 2, do a 30-minute spike: create one Job, create one Endpoint from a checkpoint, time it end-to-end. If it's multiple minutes, "pause-and-prompt" will feel broken no matter how good the UI is, and that needs to shape the UX (e.g., a "warming up" state) before you invest further.
5. **Tier 1's in-process training loop and Tier 2/3's containerised one should share the actual model/training code** (same Python module), even though the *orchestration* around them differs completely. Don't let the Tier 1 spike fork the training logic into a separate, never-reconciled copy.

---

## 4. What Stays Out of Scope at *Every* Tier (Not Just Deferred — Genuinely Not Built for This Demo)

Cross-referenced to the main project document so nothing here is "forgotten," just intentionally not part of this build:

- Authentication, multi-user accounts, sessions (§11 of main doc — stubbed out entirely here)
- Public/private experiments, forking, template gallery (§6.1, §10 of main doc)
- Cost governance: GPU quotas, kill switches, queueing (§14 of main doc)
- Dataset governance, full reproducibility metadata (Docker tags, git commits) (§14 of main doc)
- MoE expert-usage metrics, perplexity, tokens/sec and other Phase 2 evaluation metrics (§6.2 of main doc) — fine to add later, not required to prove the core mechanism
- Stream 2 (RNN/GRU/LSTM) — Tiny Transformer only, per current document scope (§5)
- MLflow/TensorBoard integration (§6.2 of main doc)
- GitHub push automation, export bundle as zip (§6.2 of main doc) — Tier 1-3 only need single-notebook download

---

## 5. Recommended Build Order

1. **Build Tier 1 first, completely, and demo it.** This is where the actual product idea — config UI, code view, pause-and-prompt, loss curve — gets proven or disproven. Nothing about Nebius infrastructure changes whether this *idea* works.
2. **Before starting Tier 2, do the cold-start spike (Gotcha #4) in isolation** — a throwaway script, not integrated into the app yet. This de-risks the single scariest unknown before you build the rest of Tier 2 around it.
3. **Build Tier 2**, swapping the orchestration layer under the same UI.
4. **Build Tier 3 last** — it's additive and low-risk relative to the other two.

---

*This document is scoped for prototype/demo purposes only. For product positioning, full feature set, and longer-term architecture, see `LLM_Experiments_Lab_Project_Discussion.md`.*
