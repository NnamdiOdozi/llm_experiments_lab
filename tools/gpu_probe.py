#!/usr/bin/env python3
"""GPU training probe — measures backend + system responsiveness during GPU training.

Uses REAL preset model sizes (not toy models).
Monitors nvidia-smi GPU utilization alongside API polling.
Tests pause/resume/stop under GPU load.

Usage:
    # Start backend first:  cd llm_experiments_lab && uv run uvicorn backend.main:app
    # Then:                 .venv/bin/python3 tools/gpu_probe.py
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import httpx

from tools.compat import find_nvidia_smi

# Real preset configs matching presets.py
CONFIGS = {
    "transformer": {
        "template": "transformer",
        "model": {
            "vocab_size": 65, "block_size": 128, "n_embd": 192,
            "n_head": 6, "n_layer": 4, "dropout": 0.1,
            "pos_encoding": "learned", "activation": "gelu",
        },
        "training": {
            "batch_size": 64, "learning_rate": 3e-4, "max_iters": 300,
            "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
        },
        "inference": {"max_new_tokens": 10, "temperature": 0.8},
        "dataset": "tiny_shakespeare",
    },
    "moe": {
        "template": "moe",
        "model": {
            "vocab_size": 65, "block_size": 128, "n_embd": 192,
            "n_head": 6, "n_layer": 4, "dropout": 0.1,
            "pos_encoding": "rope", "activation": "gelu",
            "num_experts": 8, "top_k": 2, "capacity_factor": 1.25,
        },
        "training": {
            "batch_size": 64, "learning_rate": 3e-4, "max_iters": 300,
            "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
        },
        "inference": {"max_new_tokens": 10, "temperature": 0.8},
        "dataset": "tiny_shakespeare",
    },
}


def gpu_stats() -> dict:
    """Snapshot from nvidia-smi: utilization, memory, temperature."""
    try:
        out = subprocess.check_output([
            find_nvidia_smi(),
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], timeout=5, stderr=subprocess.DEVNULL).decode().strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_util": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
            "power_w": float(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


def system_responsiveness() -> float:
    """Measure how long a trivial subprocess takes (proxy for system starvation)."""
    t0 = time.time()
    subprocess.run(["echo", "ping"], capture_output=True, timeout=5)
    return (time.time() - t0) * 1000


def timed_get(url: str) -> tuple[dict, float]:
    t0 = time.time()
    r = httpx.get(url, timeout=15)
    elapsed_ms = (time.time() - t0) * 1000
    r.raise_for_status()
    return r.json(), elapsed_ms


def run_probe(base: str, template: str):
    config = CONFIGS[template]
    print(f"GPU Probe: {template} model, target={base}")
    print(f"  Model: n_embd={config['model']['n_embd']}, n_layer={config['model']['n_layer']}, "
          f"batch_size={config['training']['batch_size']}, max_iters={config['training']['max_iters']}")

    # Baseline measurements before training
    print("\n  === Baseline (no training) ===")
    gs = gpu_stats()
    sr = system_responsiveness()
    print(f"    GPU: {gs.get('gpu_util', '?')}% util, {gs.get('mem_used_mb', '?')}MB, "
          f"{gs.get('temp_c', '?')}C, {gs.get('power_w', '?')}W")
    print(f"    System responsiveness: {sr:.0f}ms")
    _, api_lat = timed_get(f"{base}/api/experiments")
    print(f"    API baseline latency: {api_lat:.0f}ms")

    # Create experiment
    r = httpx.post(f"{base}/api/experiments", json={
        "name": f"__gpu_probe_{template}__",
        "config": config,
        "preset_key": None,
    })
    assert r.status_code == 200, f"Create failed: {r.text}"
    exp_id = r.json()["id"]

    # Start GPU training
    r = httpx.post(f"{base}/api/training/start", json={
        "experiment_id": exp_id, "device": "cuda",
    })
    assert r.status_code == 200, f"Start failed: {r.text}"
    run_id = r.json()["run_id"]
    print(f"\n  Started run {run_id} (experiment {exp_id})")

    # Wait for running
    for _ in range(60):
        st, _ = timed_get(f"{base}/api/training/{run_id}/status")
        if st.get("status") == "running":
            break
        time.sleep(0.5)
    assert st.get("status") == "running", f"Never reached running: {st}"

    # Collect GPU samples in background thread
    gpu_samples = []
    stop_sampling = threading.Event()

    def sample_gpu():
        while not stop_sampling.is_set():
            gpu_samples.append(gpu_stats())
            time.sleep(1.0)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()

    # Timed polling
    print(f"\n  === Training monitoring (500ms poll, up to 120s) ===")
    print(f"  {'poll':>4s}  {'status':12s}  {'step':>5s}  {'api_ms':>6s}  "
          f"{'sys_ms':>6s}  {'gpu%':>4s}  {'mem_MB':>6s}  {'temp':>4s}  {'power':>6s}")
    print(f"  {'----':>4s}  {'------':12s}  {'-----':>5s}  {'------':>6s}  "
          f"{'------':>6s}  {'----':>4s}  {'------':>6s}  {'----':>4s}  {'------':>6s}")

    poll_latencies = []
    sys_latencies = []
    step_history = []
    poll_start = time.time()
    last_step = -1
    poll_num = 0

    while time.time() - poll_start < 120:
        poll_num += 1
        st, api_lat = timed_get(f"{base}/api/training/{run_id}/status")
        sr = system_responsiveness()
        gs = gpu_stats()
        status = st.get("status", "?")
        step = st.get("current_step", -1)

        poll_latencies.append(api_lat)
        sys_latencies.append(sr)

        step_changed = step != last_step
        if step_changed or poll_num % 5 == 0 or api_lat > 200 or sr > 100:
            gpu_u = str(gs.get("gpu_util", "?"))
            mem_u = str(gs.get("mem_used_mb", "?"))
            temp = str(gs.get("temp_c", "?"))
            pwr = str(gs.get("power_w", "?"))
            warn = ""
            if api_lat > 500:
                warn += " !!API_SLOW"
            if sr > 100:
                warn += " !!SYS_SLOW"
            print(f"  {poll_num:4d}  {status:12s}  {step:5d}  {api_lat:6.0f}  "
                  f"{sr:6.0f}  {gpu_u:>4s}  {mem_u:>6s}  {temp:>4s}  {pwr:>6s}{warn}")
            last_step = step

        if step_changed:
            step_history.append((time.time() - poll_start, step))

        if status in ("completed", "failed", "cancelled"):
            break

        time.sleep(0.5)

    stop_sampling.set()
    sampler.join(timeout=3)

    # Summaries
    total_elapsed = time.time() - poll_start
    print(f"\n  === Summary ({template}) ===")

    print(f"\n  API Latency:")
    print(f"    Min={min(poll_latencies):.0f}ms  Med={statistics.median(poll_latencies):.0f}ms  "
          f"P95={sorted(poll_latencies)[int(len(poll_latencies)*0.95)]:.0f}ms  "
          f"Max={max(poll_latencies):.0f}ms")
    print(f"    >500ms: {sum(1 for l in poll_latencies if l > 500)}/{len(poll_latencies)}")

    print(f"\n  System Responsiveness (subprocess echo):")
    print(f"    Min={min(sys_latencies):.0f}ms  Med={statistics.median(sys_latencies):.0f}ms  "
          f"P95={sorted(sys_latencies)[int(len(sys_latencies)*0.95)]:.0f}ms  "
          f"Max={max(sys_latencies):.0f}ms")
    print(f"    >100ms: {sum(1 for l in sys_latencies if l > 100)}/{len(sys_latencies)}")

    if len(step_history) >= 2:
        first_t, first_s = step_history[0]
        last_t, last_s = step_history[-1]
        el = last_t - first_t
        if el > 0:
            print(f"\n  Step Progression: {first_s}→{last_s} ({last_s-first_s} steps in {el:.1f}s = {(last_s-first_s)/el:.1f} steps/sec)")
    else:
        print(f"\n  Step Progression: STALE — only {len(step_history)} changes in {total_elapsed:.0f}s")

    if gpu_samples:
        utils = [s["gpu_util"] for s in gpu_samples if "gpu_util" in s]
        temps = [s["temp_c"] for s in gpu_samples if "temp_c" in s]
        pwrs = [s["power_w"] for s in gpu_samples if "power_w" in s]
        if utils:
            print(f"\n  GPU Utilization: Min={min(utils)}%  Mean={statistics.mean(utils):.0f}%  Max={max(utils)}%")
        if temps:
            print(f"  GPU Temperature: Min={min(temps)}C  Mean={statistics.mean(temps):.0f}C  Max={max(temps)}C")
        if pwrs:
            print(f"  GPU Power:       Min={min(pwrs):.0f}W  Mean={statistics.mean(pwrs):.0f}W  Max={max(pwrs):.0f}W")

    # Cleanup
    r = httpx.get(f"{base}/api/training/{run_id}/status")
    final = r.json().get("status") if r.status_code == 200 else "unknown"
    if final not in ("completed", "failed", "cancelled"):
        httpx.post(f"{base}/api/training/{run_id}/stop")
        print(f"\n  [CLEANUP] Stopped run {run_id}")
    else:
        print(f"\n  Run {run_id} finished: {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--template", default="transformer", choices=["transformer", "moe"])
    args = parser.parse_args()
    run_probe(args.base_url, args.template)
