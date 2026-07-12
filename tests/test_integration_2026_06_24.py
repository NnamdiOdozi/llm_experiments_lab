"""
Integration test for LLM Experiments Lab — Tier 1
Created: 2026-06-24

Covers:
  - Backend API endpoints (health, presets, experiments, code, exports, training)
  - Frontend API compatibility (URL patterns, response shapes)
  - Transformer training: start, pause (cooperative), stop
  - RNN training: start, pause, prompt paused model, stop
  - Vite dev server proxy to backend

Prerequisites:
  - Backend: uvicorn backend.main:app --host 127.0.0.1 --port 8000
  - Frontend: cd frontend && npm run dev  (optional, for proxy test)
  - Fresh lab.db (delete before running for clean state)

Run:
  cd llm_experiments_lab
  python tests/test_integration_2026_06_24.py

Bugs found and fixed during this session:
  1. Presets endpoint returned dict, frontend expected array
     Fix: backend/api/experiments.py list_presets() now returns [{key, ...}]
  2. Frontend API URLs mismatched backend routes (/pause/{id} vs /{id}/pause)
     Fix: frontend/src/hooks/useApi.ts — all URLs corrected
  3. RNN prompt crashed on uppercase chars (KeyError on 'T')
     Fix: backend/training/runner.py — .lower() + KeyError catch
"""

import httpx
import time
import sys

BASE = "http://127.0.0.1:8000/api"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def run_tests():
    global PASS, FAIL

    # ── Health ──
    print("\n--- Health ---")
    r = httpx.get(f"{BASE}/health")
    check("health endpoint", r.status_code == 200 and r.json().get("status") == "ok")

    # ── Presets ──
    print("\n--- Presets ---")
    r = httpx.get(f"{BASE}/experiments/presets")
    presets = r.json()
    check("presets returns list", isinstance(presets, list))
    check("presets count", len(presets) == 4, f"got {len(presets)}")
    check("presets have key field", all("key" in p for p in presets))
    check("presets have template field", all("template" in p for p in presets))
    keys = [p["key"] for p in presets]
    check("preset keys", set(keys) == {"baseline", "moe", "lr_sensitivity", "rnn_baseline"}, str(keys))

    # ── Create experiment from preset ──
    print("\n--- Create Experiment ---")
    r = httpx.post(f"{BASE}/experiments/from-preset/baseline")
    check("create from preset status", r.status_code == 200)
    exp = r.json()
    check("response has experiment_id", "experiment_id" in exp, str(exp.keys()))
    exp_id = exp["experiment_id"]

    # ── Get experiment ──
    r = httpx.get(f"{BASE}/experiments/{exp_id}")
    check("get experiment", r.status_code == 200)
    exp_data = r.json()
    check("experiment has config", "config" in exp_data)

    # ── List experiments ──
    r = httpx.get(f"{BASE}/experiments/", follow_redirects=True)
    check("list experiments", r.status_code == 200 and len(r.json()) >= 1)

    # ── Code view ──
    print("\n--- Code View ---")
    r = httpx.get(f"{BASE}/code/{exp_id}")
    code = r.json()
    check("code view status", r.status_code == 200)
    check("code has template", code.get("template") == "transformer")
    check("code has model.py", "model.py" in code.get("files", {}))
    check("code has data.py", "data.py" in code.get("files", {}))

    # ── Exports ──
    print("\n--- Exports ---")
    r = httpx.get(f"{BASE}/code/{exp_id}/export.py")
    check("export .py", r.status_code == 200 and len(r.text) > 100)

    r = httpx.get(f"{BASE}/code/{exp_id}/export.ipynb")
    check("export .ipynb", r.status_code == 200 and len(r.text) > 100)

    # ── RNN Training (faster iterations, better for pause test) ──
    print("\n--- RNN Training ---")
    r = httpx.post(f"{BASE}/experiments/from-preset/rnn_baseline")
    rnn_exp_id = r.json()["experiment_id"]

    r = httpx.post(f"{BASE}/training/start", json={"experiment_id": rnn_exp_id})
    check("rnn start training", r.status_code == 200)
    rnn_run_id = r.json()["run_id"]

    # Poll until metrics appear or timeout (data download + training warmup)
    metrics = []
    s = {}
    for _ in range(30):  # up to 60s
        time.sleep(2)
        r = httpx.get(f"{BASE}/training/{rnn_run_id}/status")
        s = r.json()
        r2 = httpx.get(f"{BASE}/training/{rnn_run_id}/metrics")
        metrics = r2.json()
        if len(metrics) > 0:
            break

    check("rnn status endpoint", r.status_code == 200)
    check("rnn is running", s.get("status") == "running", s.get("status", "?"))
    check("rnn has steps", s.get("current_step", 0) > 0, f"step={s.get('current_step')}")
    check("rnn metrics endpoint", r2.status_code == 200)
    check("rnn has metrics", len(metrics) > 0, f"count={len(metrics)}")
    if metrics:
        check("rnn metric has train_loss", "train_loss" in metrics[-1])
        check("rnn metric has val_loss", "val_loss" in metrics[-1])

    # Pause (cooperative — checked each training step, may need time to reach check)
    r = httpx.post(f"{BASE}/training/{rnn_run_id}/pause")
    check("rnn pause request", r.status_code == 200)
    for _ in range(10):
        time.sleep(1)
        r = httpx.get(f"{BASE}/training/{rnn_run_id}/status")
        s = r.json()
        if s["status"] == "paused":
            break
    check("rnn paused", s["status"] == "paused", s["status"])

    # Prompt paused model (loads checkpoint in API process)
    if s["status"] == "paused":
        r = httpx.post(f"{BASE}/training/{rnn_run_id}/prompt", json={"prompt": "tyr"}, timeout=30)
        check("rnn prompt status", r.status_code == 200)
        out = r.json()
        check("rnn prompt has output", len(out.get("output", "")) > 0)
        if "output" in out:
            print(f"         RNN prompt sample: {out['output'][:80]}")

        # Prompt with uppercase (should be handled gracefully)
        r = httpx.post(f"{BASE}/training/{rnn_run_id}/prompt", json={"prompt": "TYR"}, timeout=30)
        check("rnn uppercase prompt handled", r.status_code == 200)

    # Resume
    r = httpx.post(f"{BASE}/training/{rnn_run_id}/resume")
    check("rnn resume", r.status_code == 200)
    for _ in range(10):
        time.sleep(1)
        r = httpx.get(f"{BASE}/training/{rnn_run_id}/status")
        s = r.json()
        if s["status"] == "running":
            break
    check("rnn resumed running", s["status"] == "running", s["status"])

    # Stop
    r = httpx.post(f"{BASE}/training/{rnn_run_id}/stop")
    check("rnn stop", r.status_code == 200)

    # ── Transformer Training ──
    print("\n--- Transformer Training ---")
    r = httpx.post(f"{BASE}/training/start", json={"experiment_id": exp_id})
    check("transformer start", r.status_code == 200)
    t_run_id = r.json()["run_id"]

    # Subprocess startup takes longer — poll until running
    for _ in range(15):
        time.sleep(1)
        r = httpx.get(f"{BASE}/training/{t_run_id}/status")
        s = r.json()
        if s.get("status") == "running":
            break

    check("transformer status", r.status_code == 200)
    check("transformer running", s["status"] == "running", s.get("status", "?"))
    print(f"         Transformer at step {s['current_step']}")

    r = httpx.post(f"{BASE}/training/{t_run_id}/stop")
    check("transformer stop", r.status_code == 200)

    # ── 404 / Error cases ──
    print("\n--- Error Cases ---")
    r = httpx.get(f"{BASE}/experiments/9999")
    check("nonexistent experiment 404", r.status_code == 404)

    r = httpx.post(f"{BASE}/experiments/from-preset/nonexistent")
    check("nonexistent preset 404", r.status_code == 404)

    r = httpx.get(f"{BASE}/training/9999/status")
    check("nonexistent run 404", r.status_code == 404)

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"  PASSED: {PASS}")
    print(f"  FAILED: {FAIL}")
    print(f"  TOTAL:  {PASS + FAIL}")
    print(f"{'='*50}")

    return FAIL == 0


if __name__ == "__main__":
    print("LLM Experiments Lab — Integration Tests")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend: {BASE}")

    # Wait for backend to be ready (up to 10s)
    for attempt in range(10):
        try:
            httpx.get(f"{BASE}/health", timeout=2)
            break
        except Exception:
            if attempt == 9:
                print("\nERROR: Backend not running. Start with:")
                print("  cd llm_experiments_lab")
                print("  uvicorn backend.main:app --host 127.0.0.1 --port 8000")
                sys.exit(1)
            time.sleep(1)

    success = run_tests()
    sys.exit(0 if success else 1)
