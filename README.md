# LLM Experiments Lab

Browser-based lab for hands-on LLM architecture experiments. Pick a preset model config, tweak hyperparameters, train on CPU, and watch loss curves update live.

## What It Does

You select from preset experiments (tiny transformers, RNNs), optionally adjust config (layers, heads, learning rate, etc.), hit "Start", and watch training happen with live loss charts. You can pause training mid-run, prompt the model with text to see current output quality, then resume or stop. When done, export your experiment as a standalone `.py` script or Jupyter notebook.

**Available presets:**

| Preset | Architecture | Dataset | Purpose |
|--------|-------------|---------|---------|
| Baseline Tiny Transformer | 4-layer decoder-only | Tiny Shakespeare | Default starting point |
| RoPE Positional Encoding | Same + rotary embeddings | Tiny Shakespeare | Compare position encodings |
| High Learning Rate | Same + 10x LR | Tiny Shakespeare | Observe training instability |
| Baseline CharRNN (LSTM) | 2-layer LSTM | Dinosaur names | RNN vs Transformer comparison |

## Architecture

- **Backend:** FastAPI + SQLite (async via aiosqlite). Runs training in-process with PyTorch. Streams metrics to frontend via polling (WebSocket endpoint also available).
- **Frontend:** React + TypeScript + Vite. Recharts for loss visualization. Polls backend every 2s during training.
- **Database:** SQLite file (`lab.db`, auto-created on first run). Stores experiments, training runs, and metrics.

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

## Setup & Run

### 1. Backend

From the `llm_experiments_lab/` directory:

```bash
# Install Python dependencies
uv sync

# Start the API server (port 8000)
uv run uvicorn backend.main:app --reload
```

Verify: `curl http://localhost:8000/api/health` should return `{"status":"ok"}`.

API docs at `http://localhost:8000/docs` (Swagger UI).

### 2. Frontend

In a second terminal, from `llm_experiments_lab/frontend/`:

```bash
# Install JS dependencies (first time only)
npm install

# Start dev server (port 5173)
npm run dev
```

Open `http://localhost:5173` in browser.

## How to Use

1. **Pick a preset** from the landing page — this creates an experiment in the database.
2. **Tweak config** in the left sidebar (model dimensions, training hyperparams).
3. **Click Start** — training begins on CPU. Loss chart updates live.
4. **Pause** mid-training to prompt the model and see its current text generation quality.
5. **Resume or Stop** when ready.
6. **Export** your experiment as a `.py` script or `.ipynb` notebook via the export bar.
7. **View code** — the actual model and data source files are shown in the Code View panel.

## API Endpoints

| Method | Path | What |
|--------|------|------|
| GET | `/api/health` | Health check |
| GET | `/api/experiments` | List all experiments |
| POST | `/api/experiments/from-preset/{key}` | Create from preset |
| GET | `/api/experiments/presets` | List available presets |
| POST | `/api/training/start` | Start training run |
| POST | `/api/training/{id}/pause` | Pause run |
| POST | `/api/training/{id}/resume` | Resume run |
| POST | `/api/training/{id}/prompt` | Prompt paused model |
| GET | `/api/code/{id}/export.py` | Download .py script |
| GET | `/api/code/{id}/export.ipynb` | Download notebook |

## Project Structure

```
llm_experiments_lab/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── db.py                # SQLite operations
│   ├── api/                 # Route handlers (experiments, training, codegen)
│   ├── training/
│   │   ├── runner.py        # Training loop orchestration
│   │   └── templates/       # Model code (transformer/, rnn/)
│   └── export.py            # Script/notebook generation
├── frontend/
│   └── src/
│       ├── App.tsx           # Main layout + state management
│       └── components/       # UI panels (config, chart, controls, code view)
├── config/
│   └── presets.py            # Hardcoded experiment presets
├── lab.db                    # SQLite database (auto-created)
└── pyproject.toml            # Python dependencies
```

## Current Status

Working: preset selection, experiment creation, training with live metrics, pause/resume/stop, model prompting during pause, code view, export to .py and .ipynb.

Training runs on CPU only. No GPU/auth/multi-user support yet.
