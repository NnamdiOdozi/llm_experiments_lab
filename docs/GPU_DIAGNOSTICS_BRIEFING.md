# GPU Diagnostics Briefing — Windows Native

**For:** Claude agent running on Windows native
**Context:** This project runs GPU training via a FastAPI backend. On WSL2, GPU training freezes the entire system (display compositor starvation via GPU-PV). We moved to Windows native to bypass GPU-PV — but freezing may still occur. Your job: diagnose GPU behavior on Windows native.

---

## 1. Background

### What Happened on WSL2
- Training on GPU (RTX PRO 2000 Blackwell, 8GB VRAM) caused complete system freeze
- Dashboard step counter stuck at 0, then jumped 400+ steps when training was paused
- `nvidia-smi` hung for 30+ seconds during training
- Root cause: WSL2 GPU-PV (paravirtualization) serializes ALL GPU access through one bottleneck — CUDA training starves the display compositor

### Why We Moved to Windows Native
- Windows native has direct GPU driver access — no GPU-PV
- CUDA and display use separate driver paths
- Should eliminate the freeze... but we need to verify

### Current Concern
The user reports the terminal still froze on Windows. We need to determine:
1. Is the dashboard responsive during GPU training?
2. Does `nvidia-smi` respond during training?
3. Do step counters update in real-time?
4. Is this the same GPU-PV issue or something different?

---

## 2. Project Setup

### Prerequisites
```bash
cd C:\Users\nnamd\projects\llm_experiments_lab
git pull origin main    # ensure latest code with cross-platform fixes
```

### Start Servers
Terminal 1 (backend):
```bash
cd C:\Users\nnamd\projects\llm_experiments_lab
.venv\Scripts\uvicorn backend.main:app --port 8000
```

Terminal 2 (frontend):
```bash
cd C:\Users\nnamd\projects\llm_experiments_lab\frontend
npm run dev
```

Dashboard: http://localhost:5173

---

## 3. Diagnostic Tests (Run in Order)

### Test A: Baseline — Is nvidia-smi Working?

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
```

Expected: instant response with GPU stats. If this hangs, there's a driver issue unrelated to training.

### Test B: API Health Check

```bash
curl http://localhost:8000/api/experiments/presets
```

Expected: JSON array with 4 presets. If this fails, backend isn't running.

### Test C: Start GPU Training via API + Monitor

This is the critical test. Start training and repeatedly poll nvidia-smi and the API during training:

```python
# Save as test_gpu_windows.py and run with: .venv\Scripts\python test_gpu_windows.py
import subprocess
import time
import httpx

BASE = "http://localhost:8000"

CONFIG = {
    "template": "transformer",
    "model": {
        "vocab_size": 65, "block_size": 128, "n_embd": 192,
        "n_head": 6, "n_layer": 4, "dropout": 0.1,
        "pos_encoding": "learned", "activation": "gelu",
    },
    "training": {
        "batch_size": 64, "learning_rate": 3e-4, "max_iters": 200,
        "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
    },
    "inference": {"max_new_tokens": 10, "temperature": 0.8},
    "dataset": "tiny_shakespeare",
}


def nvidia_smi_latency(timeout=5):
    """Time how long nvidia-smi takes to respond."""
    try:
        t0 = time.time()
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=timeout, stderr=subprocess.DEVNULL,
        )
        return time.time() - t0
    except subprocess.TimeoutExpired:
        return None  # hung


def main():
    # Baseline
    t = nvidia_smi_latency()
    print(f"Baseline nvidia-smi: {t*1000:.0f}ms" if t else "Baseline: HUNG")

    # Create experiment
    r = httpx.post(f"{BASE}/api/experiments", json={
        "name": "__win_gpu_diag__", "config": CONFIG, "preset_key": None,
    })
    exp_id = r.json()["id"]

    # Start training on GPU
    r = httpx.post(f"{BASE}/api/training/start", json={
        "experiment_id": exp_id, "device": "cuda",
    })
    run_id = r.json()["run_id"]
    print(f"Started run {run_id}")

    # Wait for running status
    for _ in range(30):
        st = httpx.get(f"{BASE}/api/training/{run_id}/status").json()
        if st.get("status") == "running":
            break
        time.sleep(0.5)

    print(f"Status: {st.get('status')}")
    if st.get("status") != "running":
        print(f"ERROR: never reached running: {st}")
        return

    # Monitor for 60 seconds
    print(f"\n{'sec':>4s}  {'step':>5s}  {'status':10s}  {'api_ms':>7s}  {'smi_ms':>7s}  {'note'}")
    print("-" * 60)

    t0 = time.time()
    prev_step = -1
    stale_count = 0

    while time.time() - t0 < 60:
        # Time the API call
        api_t0 = time.time()
        try:
            st = httpx.get(f"{BASE}/api/training/{run_id}/status", timeout=10).json()
            api_ms = (time.time() - api_t0) * 1000
        except Exception as e:
            print(f"{time.time()-t0:4.0f}  {'?':>5s}  {'?':10s}  {'FAIL':>7s}  {'?':>7s}  API error: {e}")
            time.sleep(2)
            continue

        step = st.get("current_step", -1)
        status = st.get("status", "?")

        # Time nvidia-smi
        smi_t = nvidia_smi_latency(timeout=5)
        smi_ms = f"{smi_t*1000:.0f}" if smi_t else "HUNG"

        # Detect stale steps
        note = ""
        if step == prev_step:
            stale_count += 1
            if stale_count >= 3:
                note = "STALE STEP COUNTER"
        else:
            stale_count = 0
        if api_ms > 1000:
            note += " SLOW_API"
        if smi_t and smi_t > 1:
            note += " SLOW_SMI"
        if smi_t is None:
            note += " SMI_HUNG"

        prev_step = step

        print(f"{time.time()-t0:4.0f}  {step:5d}  {status:10s}  {api_ms:7.0f}  {smi_ms:>7s}  {note}")

        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(2)

    # Summary
    print(f"\nFinal: status={st.get('status')}, step={st.get('current_step')}")

    # Cleanup
    if st.get("status") not in ("completed", "failed", "cancelled"):
        httpx.post(f"{BASE}/api/training/{run_id}/stop")
        print("Stopped run")


if __name__ == "__main__":
    main()
```

### What to Look For in Test C Results

| Symptom | Diagnosis |
|---------|-----------|
| nvidia-smi responds <100ms during training | **GOOD** — no GPU contention |
| nvidia-smi >1s or HUNG during training | **BAD** — GPU driver contention (same as WSL2) |
| Step counter updates every 2-4 seconds | **GOOD** — yield + progress updates working |
| Step stuck at 0 for >10s then jumps | **BAD** — progress not being written or compositor starved |
| API responds <500ms during training | **GOOD** — backend not blocked |
| API >2s or FAIL during training | **BAD** — possible GIL or I/O contention |

### Test D: Dashboard Visual Check

While Test C runs (or start a separate training from the dashboard):
1. Open http://localhost:5173 in browser
2. Click a preset (e.g., "Baseline Tiny Transformer")
3. Start training on GPU
4. Watch for:
   - Does the step counter increment every few seconds?
   - Does the loss chart update?
   - Can you click Pause during training? Does it respond within 5s?
   - Does Stop work?

### Test E: Compare CPU vs GPU

Run the same training on CPU to establish a control:
```bash
curl -X POST http://localhost:8000/api/training/start -H "Content-Type: application/json" -d "{\"experiment_id\": 1, \"device\": \"cpu\"}"
```
If CPU training keeps the dashboard responsive but GPU doesn't, the problem is GPU-specific.

---

## 4. Key Files to Inspect if Issues Found

| File | What It Does | What to Check |
|------|-------------|---------------|
| `backend/training/train_worker.py` | Training loop | Look for `yield_gpu()` and `maybe_update_progress()` — these should fire regularly |
| `config/settings.py` | Yield settings | `gpu_yield_enabled`, `gpu_yield_interval_sec`, `gpu_yield_sleep` |
| `backend/training/runner.py` | Subprocess launcher | `_popen_kwargs()` — should use `CREATE_NEW_PROCESS_GROUP` on Windows |
| `tools/compat.py` | Platform helpers | `find_nvidia_smi()` should find it in PATH on Windows |

### Settings to Try if GPU Freezes

In `config/settings.py`, try increasing yield aggressiveness:
```python
gpu_yield_enabled: bool = True
gpu_yield_interval_sec: float = 0.05   # yield every 50ms (was 100ms)
gpu_yield_sleep: float = 0.030         # sleep 30ms (was 20ms)
```

---

## 5. A/B Comparison: Current Code vs Pre-Fix Baseline (`563b84a`)

On WSL2, we tested both the current code and the pre-fix commit `563b84a` — GPU behavior was similar on both (WSL2 GPU-PV dominated). On Windows native, the comparison may reveal whether the yield/progress fixes actually help.

### What Changed Since `563b84a`

Three commits modified the GPU-relevant files:

| Commit | Files Changed | What It Did |
|--------|--------------|-------------|
| `d8770fc` | `train_worker.py`, `settings.py` | Added `yield_gpu()` with configurable interval/sleep, `maybe_update_progress()` for throttled status writes |
| `953e7e5` | `train_worker.py` | Further yield tuning |
| `641c339` | `runner.py`, tools | Cross-platform `_popen_kwargs()`, nvidia-smi path fixes |

### How to Run the A/B Test

**Step 1 — Test current code (already done in Tests A-E above)**

Save the Test C output as `results_current.txt`.

**Step 2 — Checkout the old code for the two key files**

```bash
# Create a branch so you can easily switch back
git checkout -b test-pre-fix-baseline

# Checkout ONLY the GPU-relevant files from 563b84a
git checkout 563b84a -- backend/training/train_worker.py
git checkout 563b84a -- config/settings.py

# NOTE: Do NOT checkout runner.py from 563b84a — it has preexec_fn which crashes on Windows.
# The old train_worker.py + settings.py are what we want to compare (different yield behavior).
```

**Step 3 — Restart backend, run Test C again**

```bash
# Restart backend with old code
# (kill existing uvicorn, then start fresh)
.venv\Scripts\uvicorn backend.main:app --port 8000
```

Run the `test_gpu_windows.py` script again. Save output as `results_563b84a.txt`.

**Step 4 — Compare and restore**

```bash
# Compare results
# Look for differences in: nvidia-smi latency, step update frequency, API latency

# Restore the two files back to main (discard the old versions)
git checkout main -- backend/training/train_worker.py config/settings.py
git checkout main
git branch -D test-pre-fix-baseline
```

**IMPORTANT:** Only `train_worker.py` and `settings.py` change between runs. Everything else — `runner.py`, `uv.lock`, `pyproject.toml`, tools, frontend — stays on `main` throughout. Do NOT do a full `git checkout 563b84a` as that would break Windows deps and the cross-platform runner fix.

### What to Compare

| Metric | Current Code | 563b84a | Interpretation |
|--------|-------------|---------|----------------|
| nvidia-smi latency | ___ ms | ___ ms | Lower = less GPU contention |
| Step update frequency | every ___s | every ___s | More frequent = yield working |
| API latency range | ___-___ ms | ___-___ ms | Lower = backend more responsive |
| Dashboard freezes? | yes/no | yes/no | If both freeze → not a code issue |

**Key question:** If the current code (with yield fixes) performs the same as `563b84a` on Windows, then the yield mechanism isn't the bottleneck — something else is causing the freeze.

---

## 6. Report Template

After running diagnostics, report:

```
## Windows GPU Diagnostic Results

**Date:** YYYY-MM-DD
**GPU:** (output of nvidia-smi --query-gpu=name --format=csv,noheader)
**Driver:** (output of nvidia-smi --query-gpu=driver_version --format=csv,noheader)
**Python:** (python --version)
**PyTorch:** (python -c "import torch; print(torch.__version__, torch.cuda.is_available())")

### Test A — nvidia-smi baseline: [instant / slow / hung]
### Test B — API health: [ok / failed]
### Test C — During GPU training:
  - nvidia-smi latency: [range, e.g. 50-200ms / HUNG]
  - Step counter: [updating every Xs / stale]
  - API latency: [range]
  - (paste the monitoring output)
### Test D — Dashboard:
  - Step counter: [live / stuck]
  - Charts: [updating / frozen]
  - Pause/Stop: [responsive / unresponsive]
### Test E — CPU comparison: [responsive / same issue]

### Conclusion: [GPU-PV still present / different issue / resolved]
```

---

## 6. If GPU Freeze Confirmed on Windows Native

This would mean the problem is NOT GPU-PV specific. Possible alternative causes:
1. **PyTorch CUDA kernel monopolizing GPU** — try `CUDA_DEVICE_MAX_CONNECTIONS=1`
2. **Windows WDDM driver model** — TDR (Timeout Detection & Recovery) may be interfering
3. **Power management** — GPU may be throttling; check `nvidia-smi -q -d PERFORMANCE`
4. **torch.cuda.synchronize() not yielding enough** — try longer sleep in yield

Escalation: check `docs/DESIGN_DECISIONS.md` for full architectural context.
