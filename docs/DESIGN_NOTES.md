# Design Notes: Known Issues & Architecture Decisions

## GPU Training + Web UI Responsiveness (GIL Problem)

### The Problem

When training on GPU through the web UI, the dashboard may become unresponsive:
- Step counter stops updating
- Loss curves don't appear
- Pause/Stop buttons don't respond
- In severe cases, the entire terminal/VS Code freezes

This does **not** happen when running training standalone (e.g. `gpu_diag.py`).

### Root Cause: Python's Global Interpreter Lock (GIL)

The web app runs training in a background `threading.Thread`, while FastAPI handles HTTP requests in the main thread. Python's GIL only allows one thread to execute Python code at a time.

During training:
- PyTorch releases the GIL during CUDA kernel execution (the actual GPU math)
- But between kernels — batch loading, tensor indexing, Python loops, `loss.item()` — the training thread **holds the GIL**
- During eval phases (tight loop of forward passes), GIL is held almost continuously
- FastAPI's event loop can't process status/metrics requests
- UI appears frozen

This is a fundamental limitation of Python threading for CPU-bound work. It's the same reason Ctrl+C can feel slow to interrupt training in Colab or Jupyter.

### Current Mitigation

We add `time.sleep(0)` after each training step and inside eval loops. This explicitly yields the GIL, giving FastAPI a chance to process requests. It adds negligible overhead (~0.1%) but keeps the UI responsive enough for status updates and pause/stop to work.

### Proper Fix (Future Work)

Replace `threading.Thread` with `multiprocessing.Process` for training runs. A separate process has its own GIL, so training and the web server never contend. This requires:

1. Inter-process communication for metrics (shared memory, pipes, or writing to DB)
2. Process management for pause/resume (signals instead of threading.Event)
3. Careful GPU memory cleanup when stopping a process

Estimated effort: ~2 days of refactoring `runner.py` and `training.py` API.

Alternative: Run training as a subprocess (`subprocess.Popen`) executing a standalone script. Read metrics from the JSONL file on disk. Simpler but less integrated.

### What Collaborators Need to Know

1. **CPU mode works reliably** — GIL contention is minimal because training is slower and yields naturally
2. **GPU mode works but UI may lag** — training completes correctly, metrics are saved, but dashboard updates may be delayed
3. **Don't panic if UI freezes during GPU training** — the training IS running. Wait for it to finish or use `pkill -f uvicorn` from another terminal
4. **Standalone script for serious GPU runs** — use `gpu_diag.py` as a template for GPU training without the web UI
5. **Never use `--reload` with GPU training** — hot-reload kills the training process and may leave GPU memory in a bad state

### Diagnostic Results (2026-06-26)

Standalone GPU training on RTX PRO 2000 Blackwell (8GB VRAM):
- 2500 steps completed in 4 minutes (95 ms/step)
- VRAM: 92 MB steady, zero memory leak
- Loss: train 4.2 -> 1.32, val 4.2 -> 1.56
- See `gpu_diag_results.txt` for full log

## Training Speed Reference

| Device | Steps/min | Time for 5000 steps |
|--------|-----------|-------------------|
| CPU (i7/Ryzen) | ~25-50 | 100-200 min |
| RTX PRO 2000 (8GB) | ~630 | ~8 min |
| Cloud GPU (A100) | ~3000+ | <2 min |

## Eval Configuration

Default eval settings are tuned for CPU responsiveness:
- `eval_interval`: 100 steps (how often to compute validation loss)
- `eval_iters`: 10 (number of batches per eval — capped at 10 in runner.py)

Lower `eval_iters` = noisier loss estimates but faster eval. For this tiny model, 10 iters gives sufficient signal.

## File Layout

See README.md for project structure and setup instructions.
