#!/usr/bin/env python3
"""Test if CUDA_DEVICE_MAX_CONNECTIONS or power limit can prevent system starvation.

Tests:
1. Baseline — no throttle
2. CUDA_DEVICE_MAX_CONNECTIONS=1 — limit concurrent kernel launches
3. torch.cuda.set_per_process_memory_fraction — limit VRAM
4. Reduced batch size — less GPU work per step
5. nvidia-smi power limit — clock throttle

For each, runs a short training burst and checks if nvidia-smi responds.
"""

import os
import subprocess
import sys
import time
import httpx

from tools.compat import find_nvidia_smi

BASE = "http://localhost:8000"

CONFIG = {
    "template": "transformer",
    "model": {
        "vocab_size": 65, "block_size": 128, "n_embd": 192,
        "n_head": 6, "n_layer": 4, "dropout": 0.1,
        "pos_encoding": "learned", "activation": "gelu",
    },
    "training": {
        "batch_size": 64, "learning_rate": 3e-4, "max_iters": 100,
        "eval_interval": 50, "eval_iters": 2, "optimizer": "adamw",
    },
    "inference": {"max_new_tokens": 10, "temperature": 0.8},
    "dataset": "tiny_shakespeare",
}


def nvidia_smi_responsive(timeout=3):
    """Test if nvidia-smi responds within timeout."""
    try:
        t0 = time.time()
        subprocess.check_output(
            [find_nvidia_smi(), "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=timeout, stderr=subprocess.DEVNULL,
        )
        return time.time() - t0
    except subprocess.TimeoutExpired:
        return None  # hung


def run_training_and_test(label):
    """Start training, wait until running, test nvidia-smi 5 times during training."""
    print(f"\n--- {label} ---")

    r = httpx.post(f"{BASE}/api/experiments", json={
        "name": f"__throttle_{label[:10]}__", "config": CONFIG, "preset_key": None,
    })
    exp_id = r.json()["id"]
    r = httpx.post(f"{BASE}/api/training/start", json={
        "experiment_id": exp_id, "device": "cuda",
    })
    run_id = r.json()["run_id"]

    # Wait for running
    for _ in range(30):
        st = httpx.get(f"{BASE}/api/training/{run_id}/status").json()
        if st.get("status") == "running":
            break
        time.sleep(0.5)

    if st.get("status") != "running":
        print(f"  Never reached running: {st}")
        return

    # Test nvidia-smi 5 times during training
    results = []
    for i in range(5):
        t = nvidia_smi_responsive(timeout=3)
        if t is None:
            results.append("HUNG")
            print(f"  nvidia-smi test {i+1}: HUNG (>3s)")
        else:
            results.append(f"{t*1000:.0f}ms")
            print(f"  nvidia-smi test {i+1}: {t*1000:.0f}ms")
        time.sleep(1)

    # Check final status
    st = httpx.get(f"{BASE}/api/training/{run_id}/status").json()
    print(f"  Final: status={st.get('status')}, step={st.get('current_step')}")
    print(f"  nvidia-smi: {results}")

    # Cleanup
    if st.get("status") not in ("completed", "failed", "cancelled"):
        httpx.post(f"{BASE}/api/training/{run_id}/stop")

    # Wait for GPU to clear
    time.sleep(3)


def main():
    # Check backend reachable
    r = httpx.get(f"{BASE}/api/experiments")
    assert r.status_code == 200, "Backend not running"

    # Baseline nvidia-smi
    t = nvidia_smi_responsive()
    print(f"Baseline nvidia-smi: {t*1000:.0f}ms" if t else "Baseline nvidia-smi: HUNG")

    # Test 1: Current settings (whatever is configured)
    run_training_and_test("current-yield-settings")

    print("\n" + "=" * 60)
    print("NOTE: Tests that require env vars or power limits need the")
    print("training WORKER to see the changes. Since workers are spawned")
    print("by the backend, we'd need to restart backend with new env vars.")
    print("=" * 60)


if __name__ == "__main__":
    main()
