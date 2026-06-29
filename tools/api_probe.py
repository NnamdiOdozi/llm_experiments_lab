#!/usr/bin/env python3
"""Quick API probe — exercises experiment + training lifecycle to verify refactored code.

Usage:
    # Start backend first:  cd llm_experiments_lab && uvicorn backend.main:app --port 8000
    # Then:                 python tools/api_probe.py [--base-url http://localhost:8000]

Tests: create experiment → start training (CPU) → poll status → pause → resume → stop → cleanup.
"""

import argparse
import sys
import time

import httpx as requests  # httpx has requests-compatible API

PROBE_CONFIG = {
    "template": "transformer",
    "model": {
        "vocab_size": 65,
        "block_size": 32,
        "n_embd": 32,
        "n_head": 2,
        "n_layer": 1,
        "dropout": 0.0,
        "pos_encoding": "learned",
        "activation": "gelu",
    },
    "training": {
        "batch_size": 8,
        "learning_rate": 3e-4,
        "max_iters": 200,
        "eval_interval": 10,
        "eval_iters": 1,
        "optimizer": "adamw",
    },
    "inference": {"max_new_tokens": 10, "temperature": 0.8},
    "dataset": "tiny_shakespeare",
}

# Valid status values — must match backend/training/status.py RunStatus enum
VALID_STATUSES = {
    "queued", "starting", "running", "pause_requested",
    "checkpointing", "paused", "resuming", "completed", "failed", "cancelled",
}


class ProbeFailure(Exception):
    """Raised on check failure — caught by cleanup handler."""


def check(label: str, ok: bool, detail: str = ""):
    symbol = "PASS" if ok else "FAIL"
    msg = f"  [{symbol}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not ok:
        raise ProbeFailure(label)


def poll_until(base: str, run_id: int, target_statuses: set[str], timeout: int = 30) -> dict:
    """Poll /status until status is in target_statuses or timeout."""
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = requests.get(f"{base}/api/training/{run_id}/status")
        r.raise_for_status()
        last = r.json()
        if last.get("status") in target_statuses:
            return last
        time.sleep(0.5)
    raise ProbeFailure(f"TIMEOUT: waited {timeout}s for {target_statuses}, last={last.get('status')}")


def run_probe(base: str):
    print(f"Probe target: {base}\n")

    # 1. Health / connectivity
    r = requests.get(f"{base}/api/experiments")
    check("GET /api/experiments", r.status_code == 200, f"status={r.status_code}")

    # 2. Create experiment
    r = requests.post(f"{base}/api/experiments", json={
        "name": "__probe_test__",
        "config": PROBE_CONFIG,
        "preset_key": None,
    })
    check("POST /api/experiments (create)", r.status_code == 200, r.text[:100])
    exp_id = r.json()["id"]

    # 3. Start training
    run_id = None
    failed = False
    try:
        r = requests.post(f"{base}/api/training/start", json={
            "experiment_id": exp_id,
            "device": "cpu",
        })
        check("POST /api/training/start", r.status_code == 200, r.text[:100])
        run_id = r.json()["run_id"]
        status_val = r.json()["status"]
        check("Start response status is valid", status_val in VALID_STATUSES, f"status={status_val}")

        # 4. Poll until running
        st = poll_until(base, run_id, {"running", "completed"}, timeout=30)
        check("Run reached 'running'", st["status"] == "running", f"step={st.get('current_step')}")
        check("Status value is valid enum", st["status"] in VALID_STATUSES)

        # 5. Poll until at least 1 metric appears (eval_interval=10, takes ~10-15s on CPU)
        deadline = time.time() + 30
        metrics = []
        while time.time() < deadline:
            r = requests.get(f"{base}/api/training/{run_id}/metrics")
            metrics = r.json()
            if len(metrics) > 0:
                break
            time.sleep(1)
        check("GET /metrics", r.status_code == 200)
        check("Metrics received", len(metrics) > 0, f"count={len(metrics)}")

        # 6. Pause
        r = requests.post(f"{base}/api/training/{run_id}/pause")
        check("POST /pause", r.status_code == 200, r.text[:80])
        st = poll_until(base, run_id, {"paused"}, timeout=15)
        check("Run reached 'paused'", st["status"] == "paused")

        # 7. Status check while paused
        r = requests.get(f"{base}/api/training/{run_id}/status")
        check("GET /status while paused", r.status_code == 200)
        check("Paused status valid", r.json()["status"] in VALID_STATUSES)

        # 8. Resume
        r = requests.post(f"{base}/api/training/{run_id}/resume")
        check("POST /resume", r.status_code == 200, r.text[:80])
        st = poll_until(base, run_id, {"running", "completed"}, timeout=15)
        check("Run resumed to 'running'", st["status"] in ("running", "completed"))

        # 9. Stop
        if st["status"] == "running":
            r = requests.post(f"{base}/api/training/{run_id}/stop")
            check("POST /stop", r.status_code == 200)
            st = poll_until(base, run_id, {"cancelled", "completed"}, timeout=15)
            check("Run stopped", st["status"] in ("cancelled", "completed"), f"final={st['status']}")
        else:
            check("Run completed before stop (OK)", True)

        # 10. Final metrics check
        r = requests.get(f"{base}/api/training/{run_id}/metrics")
        check("Final metrics", r.status_code == 200, f"total={len(r.json())}")

    except ProbeFailure:
        failed = True

    finally:
        # Always stop the run so we don't leave stale workers consuming resources
        if run_id is not None:
            r = requests.get(f"{base}/api/training/{run_id}/status")
            final_status = r.json().get("status") if r.status_code == 200 else "unknown"
            if final_status not in ("completed", "failed", "cancelled"):
                r = requests.post(f"{base}/api/training/{run_id}/stop")
                print(f"  [CLEANUP] Stopped run {run_id} (was {final_status})")
            else:
                print(f"  [CLEANUP] Run {run_id} already terminal ({final_status})")

    if failed:
        print(f"\n  PROBE FAILED. (experiment id={exp_id})")
        sys.exit(1)
    else:
        print(f"\n  All checks passed. (experiment id={exp_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API probe for LLM Experiments Lab")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    run_probe(args.base_url)
