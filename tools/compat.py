"""Cross-platform helpers for diagnostic tools.

Provides platform-aware equivalents for nvidia-smi lookup, venv paths,
and process management so the same tool scripts run on Linux, WSL2,
and native Windows without modification.
"""

import os
import shutil
import subprocess
import sys
import time


def find_nvidia_smi() -> str:
    """Locate nvidia-smi binary across platforms.

    Search order:
    1. System PATH (works on native Linux/Windows with NVIDIA drivers)
    2. WSL2-specific path (/usr/lib/wsl/lib/)
    3. Bare name as fallback (lets subprocess raise FileNotFoundError)
    """
    path = shutil.which("nvidia-smi")
    if path:
        return path
    # WSL2-specific location — only check on non-Windows
    if sys.platform != "win32":
        wsl_path = "/usr/lib/wsl/lib/nvidia-smi"
        if os.path.isfile(wsl_path):
            return wsl_path
    return "nvidia-smi"


def venv_python() -> str:
    """Return path to the venv Python executable."""
    if sys.platform == "win32":
        return r".venv\Scripts\python.exe"
    return ".venv/bin/python3"


def venv_script(name: str) -> str:
    """Return path to a venv script (e.g., 'uvicorn')."""
    if sys.platform == "win32":
        return rf".venv\Scripts\{name}.exe"
    return f".venv/bin/{name}"


def kill_backend():
    """Kill any running uvicorn backend process, cross-platform."""
    if sys.platform == "win32":
        # wmic finds python processes whose command line contains 'uvicorn'
        result = subprocess.run(
            'wmic process where "CommandLine like \'%%uvicorn%%\'" get ProcessId /format:csv',
            shell=True, capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(",")
            if parts and parts[-1].strip().isdigit():
                subprocess.run(f"taskkill /F /PID {parts[-1].strip()}", shell=True,
                               capture_output=True)
    else:
        subprocess.run("kill $(pgrep -f 'uvicorn backend') 2>/dev/null", shell=True)
    time.sleep(2)
