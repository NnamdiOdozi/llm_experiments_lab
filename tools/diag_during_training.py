#!/usr/bin/env python3
"""Start GPU training, then sample system diagnostics every 2s during training.

Captures: RAM, swap, VRAM, GPU util/temp/power, CPU load, top processes.
"""

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
        "batch_size": 64, "learning_rate": 3e-4, "max_iters": 300,
        "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
    },
    "inference": {"max_new_tokens": 10, "temperature": 0.8},
    "dataset": "tiny_shakespeare",
}


def gpu_stats():
    try:
        out = subprocess.check_output([
            find_nvidia_smi(),
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], timeout=5, stderr=subprocess.DEVNULL).decode().strip()
        parts = [x.strip() for x in out.split(",")]
        return f"GPU:{parts[0]}% {parts[1]}MB/{parts[2]}MB {parts[3]}C {parts[4]}W"
    except Exception as e:
        return f"GPU:err({e})"


def ram_stats():
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["powershell", "-Command",
                 "[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1024),"
                 "[math]::Round(((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize - "
                 "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory)/1024)"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode().strip().split()
            return f"RAM:{out[1]}MB/{out[0]}MB" if len(out) >= 2 else "RAM:?"
        except Exception:
            return "RAM:?(win)"
    out = subprocess.check_output(["free", "-m"], timeout=5).decode()
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return f"RAM:{parts[2]}MB/{parts[1]}MB"
        if line.startswith("Swap:"):
            parts = line.split()
            return f"Swap:{parts[2]}MB/{parts[1]}MB"
    return "RAM:?"


def mem_stats():
    if sys.platform == "win32":
        return ram_stats()  # Windows: single RAM line, no separate swap
    out = subprocess.check_output(["free", "-m"], timeout=5).decode()
    ram = swap = ""
    for line in out.splitlines():
        if line.startswith("Mem:"):
            p = line.split()
            ram = f"RAM:{p[2]}M/{p[1]}M"
        elif line.startswith("Swap:"):
            p = line.split()
            swap = f"Swap:{p[2]}M/{p[1]}M"
    return f"{ram} {swap}"


def cpu_load():
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "LoadPercentage", "/format:csv"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.splitlines():
                parts = line.strip().split(",")
                if parts and parts[-1].strip().isdigit():
                    return f"Load:{parts[-1].strip()}%"
        except Exception:
            pass
        return "Load:?(win)"
    out = subprocess.check_output(["uptime"], timeout=5).decode().strip()
    idx = out.find("load average:")
    if idx >= 0:
        return "Load:" + out[idx + 13:].strip()
    return "Load:?"


def top_cpu_procs():
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["powershell", "-Command",
                 "Get-Process | Sort-Object CPU -Descending | Select-Object -First 3 "
                 "-Property ProcessName,CPU | Format-Table -HideTableHeaders"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode().strip()
            return out.replace("\n", " | ") if out else "?(win)"
        except Exception:
            return "?(win)"
    out = subprocess.check_output(
        ["ps", "aux", "--sort=-%cpu"], timeout=5
    ).decode().splitlines()
    procs = []
    for line in out[1:4]:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            procs.append(f"{parts[10][:30]}({parts[2]}%cpu)")
    return " | ".join(procs)


def main():
    print("=== Baseline (no training) ===")
    print(f"  {mem_stats()}  {gpu_stats()}  {cpu_load()}")

    # Create experiment + start training
    r = httpx.post(f"{BASE}/api/experiments", json={
        "name": "__diag__", "config": CONFIG, "preset_key": None,
    })
    exp_id = r.json()["id"]
    r = httpx.post(f"{BASE}/api/training/start", json={
        "experiment_id": exp_id, "device": "cuda",
    })
    run_id = r.json()["run_id"]
    print(f"\nStarted run {run_id}")

    # Wait for running
    for _ in range(30):
        st = httpx.get(f"{BASE}/api/training/{run_id}/status").json()
        if st.get("status") == "running":
            break
        time.sleep(0.5)

    print(f"\n=== During training (sampling every 2s) ===")
    print(f"{'t':>4s}  {'step':>5s}  {'memory':30s}  {'gpu':30s}  {'load':20s}")
    print("-" * 95)

    t0 = time.time()
    while True:
        st = httpx.get(f"{BASE}/api/training/{run_id}/status", timeout=10).json()
        status = st.get("status", "?")
        step = st.get("current_step", -1)
        elapsed = time.time() - t0

        mem = mem_stats()
        gpu = gpu_stats()
        load = cpu_load()

        print(f"{elapsed:4.0f}  {step:5d}  {mem:30s}  {gpu:30s}  {load:20s}")

        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(2)

    print(f"\n=== Post-training ===")
    time.sleep(2)
    print(f"  {mem_stats()}  {gpu_stats()}  {cpu_load()}")
    print(f"\n  Top CPU: {top_cpu_procs()}")

    # Cleanup
    httpx.post(f"{BASE}/api/training/{run_id}/stop", timeout=5)


if __name__ == "__main__":
    main()
