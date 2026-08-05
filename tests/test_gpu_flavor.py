"""Tests for GPU flavor selection (L40S vs H100)."""

import pytest

from backend import db
from backend.nebius import endpoints_client, worker_manager
from backend.training.worker_status import WorkerStatus, session_id_for


class TestSessionIdFor:
    """Test session_id_for() flavor parameter encoding."""

    def test_session_id_for_cpu_ignores_flavor(self):
        """CPU sessions always use 'worker-cpu' regardless of flavor."""
        assert session_id_for("cpu", None) == "worker-cpu"
        assert session_id_for("cpu", "l40s") == "worker-cpu"
        assert session_id_for("cpu", "h100") == "worker-cpu"

    def test_session_id_for_gpu_l40s_default(self):
        """GPU with L40S (default or explicit) uses 'worker-gpu'."""
        assert session_id_for("gpu") == "worker-gpu"
        assert session_id_for("gpu", None) == "worker-gpu"
        assert session_id_for("gpu", "l40s") == "worker-gpu"

    def test_session_id_for_gpu_h100(self):
        """GPU with H100 flavor uses 'worker-gpu-h100'."""
        assert session_id_for("gpu", "h100") == "worker-gpu-h100"

    def test_backward_compat_default_is_l40s(self):
        """Existing code that doesn't pass gpu_flavor defaults to L40S key."""
        # This is the backward-compatibility guarantee:
        # session_id_for("gpu") must return "worker-gpu", not a new key.
        assert session_id_for("gpu") == "worker-gpu"


class TestEndpointCreateKwargs:
    """Test endpoint_create_kwargs() flavor selection."""

    def test_cpu_ignores_flavor(self):
        """CPU endpoints use CPU settings regardless of flavor."""
        kwargs_default = worker_manager.endpoint_create_kwargs("cpu")
        kwargs_l40s = worker_manager.endpoint_create_kwargs("cpu", "l40s")
        kwargs_h100 = worker_manager.endpoint_create_kwargs("cpu", "h100")

        # All should be identical for CPU
        assert kwargs_default["name"] == kwargs_l40s["name"] == kwargs_h100["name"]
        assert kwargs_default["platform"] == kwargs_l40s["platform"] == kwargs_h100["platform"]

    def test_gpu_l40s_default(self):
        """GPU with L40S (default or explicit) uses L40S settings."""
        kwargs_default = worker_manager.endpoint_create_kwargs("gpu")
        kwargs_explicit = worker_manager.endpoint_create_kwargs("gpu", "l40s")

        # Both should match (L40S is default)
        assert kwargs_default["platform"] == kwargs_explicit["platform"]
        assert kwargs_default["preset"] == kwargs_explicit["preset"]
        assert kwargs_default["name"] == kwargs_explicit["name"]

    def test_gpu_h100_uses_h100_settings(self, monkeypatch):
        """GPU with H100 flavor uses H100-specific platform/preset/name."""
        # Mock the settings to see exactly which values are selected
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_platform", "gpu-l40s-a"
        )
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_preset", "1gpu-8vcpu-32gb"
        )
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_endpoint_name", "llm-lab-gpu-trainer"
        )
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_h100_platform", "gpu-h100-sxm"
        )
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_h100_preset", "1gpu-16vcpu-200gb"
        )
        monkeypatch.setattr(
            worker_manager.settings, "nebius_gpu_h100_endpoint_name", "llm-lab-gpu-h100-trainer"
        )

        kwargs_l40s = worker_manager.endpoint_create_kwargs("gpu", "l40s")
        kwargs_h100 = worker_manager.endpoint_create_kwargs("gpu", "h100")

        # L40S values
        assert kwargs_l40s["platform"] == "gpu-l40s-a"
        assert kwargs_l40s["preset"] == "1gpu-8vcpu-32gb"
        assert kwargs_l40s["name"] == "llm-lab-gpu-trainer"

        # H100 values
        assert kwargs_h100["platform"] == "gpu-h100-sxm"
        assert kwargs_h100["preset"] == "1gpu-16vcpu-200gb"
        assert kwargs_h100["name"] == "llm-lab-gpu-h100-trainer"

        # Both use the same GPU trainer image
        assert kwargs_l40s["image"] == kwargs_h100["image"]

    def test_gpu_none_flavor_is_l40s(self):
        """GPU with None flavor (backward compat) uses L40S settings."""
        kwargs_none = worker_manager.endpoint_create_kwargs("gpu", None)
        kwargs_l40s = worker_manager.endpoint_create_kwargs("gpu", "l40s")

        assert kwargs_none["platform"] == kwargs_l40s["platform"]
        assert kwargs_none["preset"] == kwargs_l40s["preset"]


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    """Temporary test database."""
    db_path = tmp_path / "test_gpu_flavor.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()


@pytest.fixture(autouse=True)
def mock_endpoints(monkeypatch):
    """Mock Nebius endpoints to avoid real API calls."""
    async def fake_find_endpoint_any_state(name):
        return None

    async def fake_probe(url):
        return True

    monkeypatch.setattr(endpoints_client, "find_endpoint_any_state", fake_find_endpoint_any_state)
    monkeypatch.setattr(endpoints_client, "probe_endpoint_url", fake_probe)


async def test_ensure_worker_separate_sessions_per_flavor(temp_db, monkeypatch):
    """Ensure worker creates separate endpoints for L40S and H100."""
    create_calls = []

    async def fake_create_endpoint(**kwargs):
        create_calls.append(kwargs)
        endpoint_id = f"endpoint-{len(create_calls)}"
        return endpoint_id

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://endpoint.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    # Request L40S worker
    worker_l40s = await worker_manager.ensure_worker("cuda", "l40s")
    assert worker_l40s["worker_status"] == WorkerStatus.READY

    # Request H100 worker
    worker_h100 = await worker_manager.ensure_worker("cuda", "h100")
    assert worker_h100["worker_status"] == WorkerStatus.READY

    # Both should have created separate endpoints (different names)
    assert len(create_calls) >= 2
    # The endpoint names should reflect the flavor
    names = [call["name"] for call in create_calls]
    assert "llm-lab-gpu-trainer" in names or "llm-lab-gpu-h100-trainer" in names

    # Verify sessions are separate
    session_l40s = await db.get_worker_session("worker-gpu")
    session_h100 = await db.get_worker_session("worker-gpu-h100")

    # Both should exist (separate workers)
    assert session_l40s is not None
    assert session_h100 is not None


async def test_session_id_flavor_encodes_in_db(temp_db, monkeypatch):
    """Verify that gpu_flavor is properly encoded in session_id for DB lookups."""
    # When ensure_worker("cuda", "l40s") is called, session_id is "worker-gpu"
    # When ensure_worker("cuda", "h100") is called, session_id is "worker-gpu-h100"
    # These are separate rows in the DB.

    async def fake_create_endpoint(**kwargs):
        return "endpoint-test"

    async def fake_get_endpoint(endpoint_id):
        return {
            "spec": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb"},
            "status": {"state": "RUNNING", "public_endpoints": ["https://test.tunnel.nebius.cloud"]},
        }

    monkeypatch.setattr(endpoints_client, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(endpoints_client, "get_endpoint", fake_get_endpoint)

    # Create L40S worker
    await worker_manager.ensure_worker("cuda", "l40s")
    session_l40s = await db.get_worker_session("worker-gpu")
    assert session_l40s is not None
    assert session_l40s["session_id"] == "worker-gpu"

    # Create H100 worker
    await worker_manager.ensure_worker("cuda", "h100")
    session_h100 = await db.get_worker_session("worker-gpu-h100")
    assert session_h100 is not None
    assert session_h100["session_id"] == "worker-gpu-h100"

    # Both exist independently
    assert session_l40s["id"] != session_h100["id"]
