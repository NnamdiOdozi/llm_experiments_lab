"""Tests for backend/training/train_worker.py's resource-usage sampling
(CPU/RAM via psutil, GPU via nvidia-smi) — 2026-07-12 addition, requested
after the app had no way to show utilization metrics at all.
"""
from backend.training import train_worker


class FakeVirtualMemory:
    def __init__(self, used, total):
        self.used = used
        self.total = total


def test_sample_resource_usage_includes_cpu_and_ram_for_cpu_device(monkeypatch):
    monkeypatch.setattr(train_worker.psutil, "cpu_percent", lambda interval=None: 42.5)
    monkeypatch.setattr(
        train_worker.psutil, "virtual_memory",
        lambda: FakeVirtualMemory(used=4 * 1024**3, total=16 * 1024**3),
    )

    usage = train_worker._sample_resource_usage("cpu")

    assert usage["cpu_percent"] == 42.5
    assert usage["ram_used_mb"] == 4096.0
    assert usage["ram_total_mb"] == 16384.0
    assert "gpu_utilization_pct" not in usage


def test_sample_resource_usage_includes_gpu_fields_for_cuda_device(monkeypatch):
    monkeypatch.setattr(train_worker.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(
        train_worker.psutil, "virtual_memory",
        lambda: FakeVirtualMemory(used=1024**3, total=8 * 1024**3),
    )

    def fake_run(*args, **kwargs):
        class Result:
            stdout = "87, 12000, 24576, 65\n"
        return Result()

    monkeypatch.setattr(train_worker.sp, "run", fake_run)

    usage = train_worker._sample_resource_usage("cuda")

    assert usage["gpu_utilization_pct"] == 87.0
    assert usage["gpu_memory_used_mb"] == 12000.0
    assert usage["gpu_memory_total_mb"] == 24576.0
    assert usage["gpu_temp_c"] == 65.0


def test_sample_resource_usage_is_non_fatal_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(train_worker.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(
        train_worker.psutil, "virtual_memory",
        lambda: FakeVirtualMemory(used=1024**3, total=8 * 1024**3),
    )

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(train_worker.sp, "run", fake_run)

    usage = train_worker._sample_resource_usage("cuda")

    assert "gpu_utilization_pct" not in usage
    assert usage["cpu_percent"] == 10.0


def test_sample_resource_usage_is_non_fatal_when_psutil_fails(monkeypatch):
    def fake_cpu_percent(interval=None):
        raise RuntimeError("psutil sampling failed")

    monkeypatch.setattr(train_worker.psutil, "cpu_percent", fake_cpu_percent)

    usage = train_worker._sample_resource_usage("cpu")

    assert usage == {}
