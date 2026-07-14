import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.main import app


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment",
        {
            "template": "transformer",
            "model": {"block_size": 128, "n_embd": 192},
            "training": {"learning_rate": 0.0003},
            "inference": {"max_new_tokens": 100, "temperature": 0.8, "decoding_mode": "sample"},
        },
    )
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_update_config_rejects_max_new_tokens_above_block_size(temp_db, client):
    """Direct user request, 2026-07-15: max_new_tokens must never exceed
    block_size — applies identically to transformer and MoE, since both
    share the same diagnostics/generation code (see
    docs/DESIGN_DECISIONS.md §57). Without the cap, exceeding block_size
    crashed the model mid-generation with a tensor-size mismatch."""
    resp = await client.patch(
        f"/api/experiments/{temp_db}/config",
        json={
            "config": {
                "template": "transformer",
                "model": {"block_size": 128, "n_embd": 192},
                "training": {"learning_rate": 0.0003},
                "inference": {"max_new_tokens": 150, "temperature": 0.8, "decoding_mode": "sample"},
            }
        },
    )
    assert resp.status_code == 400
    assert "150" in resp.json()["detail"]
    assert "128" in resp.json()["detail"]

    # Rejected PATCH must not have been persisted.
    exp = await db.get_experiment(temp_db)
    import json
    assert json.loads(exp["config_json"])["inference"]["max_new_tokens"] == 100


async def test_update_config_allows_max_new_tokens_equal_to_block_size(temp_db, client):
    resp = await client.patch(
        f"/api/experiments/{temp_db}/config",
        json={
            "config": {
                "template": "transformer",
                "model": {"block_size": 128, "n_embd": 192},
                "training": {"learning_rate": 0.0003},
                "inference": {"max_new_tokens": 128, "temperature": 0.8, "decoding_mode": "sample"},
            }
        },
    )
    assert resp.status_code == 200


async def test_update_config_allows_max_new_tokens_below_block_size(temp_db, client):
    resp = await client.patch(
        f"/api/experiments/{temp_db}/config",
        json={
            "config": {
                "template": "moe",
                "model": {"block_size": 128, "n_embd": 192},
                "training": {"learning_rate": 0.0003},
                "inference": {"max_new_tokens": 80, "temperature": 0.8, "decoding_mode": "sample"},
            }
        },
    )
    assert resp.status_code == 200


async def test_update_config_allows_zero_temperature(temp_db, client):
    """Direct user request, 2026-07-15: temperature=0 is allowed to save —
    clamped to a tiny epsilon at the point it's actually used for division
    (torch.softmax(logits / temperature, ...)), not rejected here. See
    test_diagnostic_sampling_clamps_zero_temperature in test_diagnostics.py
    and docs/DESIGN_DECISIONS.md."""
    resp = await client.patch(
        f"/api/experiments/{temp_db}/config",
        json={
            "config": {
                "template": "transformer",
                "model": {"block_size": 128, "n_embd": 192},
                "training": {"learning_rate": 0.0003},
                "inference": {"max_new_tokens": 100, "temperature": 0, "decoding_mode": "sample"},
            }
        },
    )
    assert resp.status_code == 200
