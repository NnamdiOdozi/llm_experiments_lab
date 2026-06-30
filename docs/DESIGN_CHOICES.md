# Design Choices

Tricky bugs and non-obvious decisions — documented to prevent regressions.

## GPU Yield: Time-Based, Not Step-Based

**Problem:** Dashboard freezes during fast GPU training on WSL2. Steps and charts
show zero updates, then jump 400-500 steps when Pause is pressed.

**Root cause:** WSL2's display compositor shares the GPU with training. The old
step-based yield (`cuda.synchronize() + sleep(1ms)` every 10 steps) was too short
for the compositor to render a frame (needs ~16.7ms at 60fps). Once the GPU hits
full speed, the browser literally cannot repaint.

**Why it looks like a freeze:** Frontend polling IS working, React IS updating
state, but the browser window cannot repaint. When Pause stops the GPU, the
compositor catches up and renders all buffered state changes at once — big step jump.

**Solution:** Time-based yield in `WorkerState.yield_gpu()`:
- Fires every `gpu_yield_interval_sec` (100ms wall time), not every N steps
- Sleeps `gpu_yield_sleep` (20ms) — enough for one compositor frame
- Also calls `update_progress()` to write fresh `current_step` to status.json
- Adapts to training speed: barely fires on slow CPU, fires regularly on fast GPU
- Overhead: ~17% (20ms/120ms) — acceptable for a lab environment

**Settings** (`config/settings.py`):
- `gpu_yield_interval_sec: float = 0.1`
- `gpu_yield_sleep: float = 0.02`

**DO NOT:** Revert to step-based yield or reduce the sleep below 16ms.
The compositor needs a full frame to render.
