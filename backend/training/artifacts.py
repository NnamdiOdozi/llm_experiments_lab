"""Run directory layout and file-based communication for training workers."""

import json
from pathlib import Path

from config.settings import settings


def run_dir(run_id: int) -> Path:
    d = settings.data_dir / "runs" / str(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path(run_id: int) -> Path:
    return run_dir(run_id) / "config.json"


def status_path(run_id: int) -> Path:
    return run_dir(run_id) / "status.json"


def metrics_path(run_id: int) -> Path:
    return run_dir(run_id) / "metrics.jsonl"


def checkpoint_path(run_id: int) -> Path:
    return run_dir(run_id) / "checkpoint.pt"


def write_status(run_id: int, status_dict: dict):
    sp = status_path(run_id)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status_dict))
    tmp.replace(sp)


def read_status(run_id: int) -> dict | None:
    p = status_path(run_id)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_flag(run_id: int, flag: str):
    (run_dir(run_id) / f"{flag}.flag").touch()


def remove_flag(run_id: int, flag: str):
    (run_dir(run_id) / f"{flag}.flag").unlink(missing_ok=True)


def has_flag(run_id: int, flag: str) -> bool:
    return (run_dir(run_id) / f"{flag}.flag").exists()
