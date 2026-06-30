#!/usr/bin/env python3
"""Sweep different GPU yield configurations and measure throughput + system impact.

Modifies settings.py, restarts backend, runs gpu_probe, collects results.
Restores original settings when done.
"""

import re
import subprocess
import sys
import time

from tools.compat import kill_backend as _kill_backend, venv_python, venv_script

SETTINGS_PATH = "config/settings.py"

# (interval_steps, sleep_seconds, label)
CONFIGS = [
    (10, 0.001, "10-step/1ms (baseline 563b84a)"),
    (5,  0.001, "5-step/1ms"),
    (1,  0.001, "1-step/1ms"),
    (1,  0.005, "1-step/5ms"),
    (1,  0.010, "1-step/10ms"),
    (1,  0.020, "1-step/20ms"),
    (1,  0.050, "1-step/50ms"),
    (5,  0.020, "5-step/20ms"),
    (5,  0.050, "5-step/50ms"),
]


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def set_yield_config(content, interval, sleep_val):
    content = re.sub(
        r'gpu_yield_interval: int = \d+',
        f'gpu_yield_interval: int = {interval}',
        content
    )
    content = re.sub(
        r'gpu_yield_sleep: float = [\d.]+',
        f'gpu_yield_sleep: float = {sleep_val}',
        content
    )
    return content


def kill_backend():
    _kill_backend()


def start_backend():
    proc = subprocess.Popen(
        [venv_script("uvicorn"), "backend.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    return proc


def run_probe(template="transformer"):
    result = subprocess.run(
        [venv_python(), "tools/gpu_probe.py", "--template", template],
        capture_output=True, text=True, timeout=180,
    )
    return result.stdout + result.stderr


def extract_rate(output):
    """Pull steps/sec from probe output."""
    match = re.search(r'([\d.]+) steps/sec', output)
    return float(match.group(1)) if match else None


def extract_api_p95(output):
    match = re.search(r'P95=(\d+)ms', output)
    return int(match.group(1)) if match else None


def extract_sys_max(output):
    match = re.search(r'System.*Max=(\d+)ms', output)
    return int(match.group(1)) if match else None


def main():
    original = read_file(SETTINGS_PATH)
    results = []

    print("=" * 90)
    print(f"{'Config':30s}  {'steps/s':>8s}  {'API P95':>8s}  {'Sys Max':>8s}  {'Overhead':>8s}")
    print("=" * 90)

    baseline_rate = None

    try:
        for interval, sleep_val, label in CONFIGS:
            # Update settings
            modified = set_yield_config(original, interval, sleep_val)
            write_file(SETTINGS_PATH, modified)

            # Restart backend
            kill_backend()
            proc = start_backend()

            try:
                output = run_probe("transformer")
                rate = extract_rate(output)
                api_p95 = extract_api_p95(output)
                sys_max = extract_sys_max(output)

                if baseline_rate is None and rate:
                    baseline_rate = rate

                overhead = ""
                if rate and baseline_rate:
                    pct = (1 - rate / baseline_rate) * 100
                    overhead = f"{pct:+.0f}%"

                print(f"  {label:28s}  {rate or '?':>8}  {api_p95 or '?':>8}  "
                      f"{sys_max or '?':>8}  {overhead:>8s}")

                results.append({
                    "label": label, "rate": rate,
                    "api_p95": api_p95, "sys_max": sys_max,
                })
            finally:
                proc.terminate()
                proc.wait(timeout=5)

    finally:
        # Restore original settings
        write_file(SETTINGS_PATH, original)
        kill_backend()
        print("\n  Settings restored to original.")

    print("\n" + "=" * 90)
    print("DONE. Pick config with best overhead-vs-responsiveness trade-off.")
    print("Note: system responsiveness measured from Python subprocess (not GUI).")
    print("GUI/compositor starvation cannot be measured here — user dashboard test needed.")


if __name__ == "__main__":
    main()
