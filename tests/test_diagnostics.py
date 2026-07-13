"""Tests for diagnostic endpoints (architecture, start, step, get).

Tests cover: architecture manifest generation, diagnostic session creation,
forward pass capture, and response validation against the contract schema.
"""

import asyncio
import json
import pytest
import httpx
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.api import training as training_module
from backend.main import app
from backend.nebius import worker_manager
from backend.training.status import RunStatus
from config.settings import settings


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://fake.example")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self):
        return self._json_data


class FakeAsyncClient:
    """Replaces httpx.AsyncClient — returns queued responses in call order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return self.responses.pop(0)


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment", {"template": "transformer"}, preset_key="tiny-shakespeare",
    )
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_get_architecture_manifest_transformer(temp_db, client, monkeypatch, tmp_path):
    """GET /api/training/{run_id}/architecture returns correct schema."""
    from backend.training import artifacts

    # Isolate config.json under tmp_path — see comment in
    # test_diagnostics_step_returns_correct_schema for why.
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    exp_id = temp_db

    # Create the run row directly (no real subprocess) — POST /training/start
    # for a local run actually spawns a real train_worker.py subprocess, which
    # this test doesn't need and which previously crashed on temp_db's
    # intentionally-minimal experiment config ({"template": "transformer"},
    # no "model"/"training" keys), leaking a stale entry into the process-wide
    # `active_runs` dict in backend/training/runner.py that isn't per-test
    # isolated — that leak broke two unrelated open-runs tests later in the
    # same pytest session. See docs/DESIGN_DECISIONS.md.
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    config = {
        "template": "transformer",
        "device": "cpu",
        "model": {
            "vocab_size": 65, "block_size": 128, "n_embd": 192, "n_head": 6,
            "n_layer": 4, "dropout": 0.1, "pos_encoding": "learned", "activation": "gelu",
        },
        "training": {
            "batch_size": 64, "learning_rate": 3e-4, "max_iters": 1000,
            "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
        },
    }
    (rd / "config.json").write_text(json.dumps(config))

    # Fetch architecture
    resp = await client.get(f"/api/training/{run_id}/architecture")
    assert resp.status_code == 200
    arch = resp.json()

    # Validate schema
    assert arch["schema_version"] == 1
    assert arch["local_run_id"] == run_id
    assert arch["template"] == "transformer"
    assert "param_count" in arch
    assert "trainable_param_count" in arch
    # Note: param count may be 0 if model build failed due to incomplete config
    # but schema structure should still be present

    # Validate nodes
    nodes = arch["nodes"]
    assert len(nodes) > 0

    # Check embedding node
    embedding_nodes = [n for n in nodes if n["id"] == "embedding"]
    assert len(embedding_nodes) == 1
    emb = embedding_nodes[0]
    assert emb["kind"] == "embedding"
    assert "config" in emb
    assert "vocab_size" in emb["config"]
    assert "n_embd" in emb["config"]

    # Check block group
    block_nodes = [n for n in nodes if n["id"] == "block"]
    assert len(block_nodes) == 1
    block = block_nodes[0]
    assert block["kind"] == "transformer_block_group"
    assert block["repeat_count"] == 4  # default n_layer
    assert len(block["children"]) > 0

    # Check children have expected structure
    child_ids = [c["id"] for c in block["children"]]
    assert "block.{i}.ln1" in child_ids
    assert "block.{i}.attention" in child_ids
    assert "block.{i}.ln2" in child_ids
    assert "block.{i}.mlp" in child_ids

    # Check final_norm and lm_head
    final_norm = [n for n in nodes if n["id"] == "final_norm"]
    lm_head = [n for n in nodes if n["id"] == "lm_head"]
    assert len(final_norm) == 1
    assert len(lm_head) == 1


async def test_diagnostics_start_requires_paused_run(temp_db, client):
    """POST /diagnostics/start rejects 400 if run not paused/completed."""
    exp_id = temp_db

    # Create the run row directly, QUEUED by default (no real subprocess —
    # see comment in test_get_architecture_manifest_transformer for why).
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")

    # Try to start diagnostics on a non-paused run
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "hello", "top_k": 5, "max_prompt_tokens": 32},
    )
    assert resp.status_code == 400
    assert "paused or completed" in resp.text.lower()


async def test_diagnostics_step_returns_correct_schema(temp_db, client, monkeypatch, tmp_path):
    """POST /diagnostics/{session_id}/step returns snapshot matching contract."""
    from backend.training import artifacts
    from pathlib import Path

    # Isolate run artifacts (checkpoint, config, status) under pytest's tmp_path —
    # artifacts.run_dir() resolves from settings.data_dir, which defaults to the
    # real project data/ directory. Without this, this test writes a real
    # multi-MB checkpoint into data/runs/999/ and never cleans it up (found via
    # `git status` showing stray data/runs/999,1000,1001 after a full suite run,
    # which also broke two unrelated open-runs tests via filesystem pollution).
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    exp_id = temp_db

    # Create a test run directory with a dummy checkpoint
    run_id = 999
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)

    # Write config (hardcoded for simplicity)
    config = {
        "template": "transformer",
        "device": "cpu",
        "model": {
            "vocab_size": 65,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
            "activation": "gelu",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "max_iters": 1000,
            "eval_interval": 20,
            "eval_iters": 2,
            "optimizer": "adamw",
        },
        "inference": {
            "max_new_tokens": 100,
            "temperature": 0.8,
        },
    }
    (rd / "config.json").write_text(json.dumps(config))

    # Create and save a dummy checkpoint
    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    cp = {
        "model_state": model.state_dict(),
        "config": config,
    }
    torch.save(cp, artifacts.checkpoint_path(run_id))

    # Write status as PAUSED
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})

    # Register run in DB
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    # Now start diagnostics
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The", "top_k": 5, "max_prompt_tokens": 32},
    )
    assert resp.status_code == 200
    start_resp = resp.json()
    session_id = start_resp["diagnostic_session_id"]
    assert "tokens" in start_resp

    # Run a step
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={},
    )
    assert resp.status_code == 200
    snapshot = resp.json()

    # Validate snapshot schema per contract
    assert snapshot["schema_version"] == 1
    assert snapshot["diagnostic_session_id"] == session_id
    assert snapshot["generation_step"] == 1
    assert "input_tokens" in snapshot
    assert "generated_token" in snapshot
    assert "nodes" in snapshot
    assert snapshot["complete"] is True

    # Phase 2: attention only computed when layer/head requested; not requested here
    assert snapshot["attention"]["available"] is False
    assert snapshot["attention"]["reason"] == "Not requested"
    # Phase 2: activation extras always computed (reuses logits_last, no new capture)
    assert snapshot["activation_summaries"]["available"] is True
    assert "top_abs_components" in snapshot["activation_summaries"]
    assert "value_slice" in snapshot["activation_summaries"]

    # Validate lm_head
    assert "lm_head" in snapshot
    lm_head = snapshot["lm_head"]
    assert "logits_shape" in lm_head
    assert "selected_position" in lm_head
    assert "top_k" in lm_head
    assert len(lm_head["top_k"]) <= 5
    assert lm_head["top_k"][0]["rank"] == 1

    # Validate captured nodes
    assert "embedding" in snapshot["nodes"]
    emb_capture = snapshot["nodes"]["embedding"]
    assert "input_shape" in emb_capture
    assert "output_shape" in emb_capture
    assert "summary" in emb_capture
    assert all(k in emb_capture["summary"] for k in ["mean", "std", "l2_norm", "min", "max"])

    # Check at least some block nodes are captured
    block_keys = [k for k in snapshot["nodes"].keys() if k.startswith("block.")]
    assert len(block_keys) > 0


async def test_diagnostics_get_returns_last_snapshot(temp_db, client, monkeypatch, tmp_path):
    """GET /diagnostics/{session_id} returns last snapshot without advancing."""
    from backend.training import artifacts
    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    run_id = 1000
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)

    config = {
        "template": "transformer",
        "device": "cpu",
        "model": {
            "vocab_size": 65,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
            "activation": "gelu",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "max_iters": 1000,
            "eval_interval": 20,
            "eval_iters": 2,
            "optimizer": "adamw",
        },
    }
    (rd / "config.json").write_text(json.dumps(config))

    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    cp = {"model_state": model.state_dict(), "config": config}
    torch.save(cp, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.COMPLETED, "current_step": 100, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.COMPLETED, device="cpu", execution_backend="local")

    # Start diagnostics
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "Test", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    # Run a step to create a snapshot
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={},
    )
    first_step = resp.json()
    first_gen_step = first_step["generation_step"]

    # GET should return the same snapshot
    resp = await client.get(f"/api/training/{run_id}/diagnostics/{session_id}")
    assert resp.status_code == 200
    retrieved = resp.json()
    assert retrieved["generation_step"] == first_gen_step
    assert retrieved["diagnostic_session_id"] == session_id

    # Verify that GET did NOT advance the step
    resp = await client.get(f"/api/training/{run_id}/diagnostics/{session_id}")
    second_get = resp.json()
    assert second_get["generation_step"] == first_gen_step


async def test_diagnostics_remote_requires_endpoint_available(temp_db, client, monkeypatch):
    """Remote diagnostics routes return 502 when endpoint is unavailable."""
    exp_id = temp_db

    # Create a "remote" run in DB with no valid worker session
    run_id = await db.create_training_run(
        exp_id, device="cpu", status=RunStatus.PAUSED, execution_backend="nebius_endpoint"
    )
    await db.update_training_run(
        run_id, remote_run_id=42, remote_endpoint_id="aiendpoint-123"
    )

    # Try to call diagnostics/start without a worker session — should get 502
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "test", "top_k": 5, "max_prompt_tokens": 32},
    )

    # Should fail because there's no worker session backing this run
    assert resp.status_code == 502


async def test_diagnostics_session_invalid_rejects_404(temp_db, client, monkeypatch, tmp_path):
    """POST /diagnostics/{session_id}/step with invalid session_id returns 404."""
    from backend.training import artifacts
    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    run_id = 1001
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)

    config = {
        "template": "transformer",
        "device": "cpu",
        "model": {
            "vocab_size": 65,
            "block_size": 128,
            "n_embd": 192,
            "n_head": 6,
            "n_layer": 4,
            "dropout": 0.1,
            "pos_encoding": "learned",
            "activation": "gelu",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "max_iters": 1000,
            "eval_interval": 20,
            "eval_iters": 2,
            "optimizer": "adamw",
        },
    }
    (rd / "config.json").write_text(json.dumps(config))

    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    cp = {"model_state": model.state_dict(), "config": config}
    torch.save(cp, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 50, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    # Try to step with a fake session_id
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/invalid-session-id/step",
        json={},
    )
    assert resp.status_code == 404


TRANSFORMER_CONFIG = {
    "template": "transformer",
    "device": "cpu",
    "model": {
        "vocab_size": 65, "block_size": 128, "n_embd": 192, "n_head": 6,
        "n_layer": 4, "dropout": 0.1, "pos_encoding": "learned", "activation": "gelu",
    },
    "training": {
        "batch_size": 64, "learning_rate": 3e-4, "max_iters": 1000,
        "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
    },
}


def test_model_generate_greedy_never_calls_multinomial():
    """model.generate(greedy=True) — the Generate button's underlying call
    — must select via torch.argmax, never torch.multinomial. Same
    decoding_mode setting used by diagnostics.py's step-through path;
    this is the direct model-level unit test for it. See
    docs/DESIGN_DECISIONS.md."""
    import torch
    from backend.training.templates import TEMPLATE_REGISTRY

    model = TEMPLATE_REGISTRY["transformer"]["build_model"](TRANSFORMER_CONFIG)
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    calls = []
    real_multinomial = torch.multinomial
    with_patch = lambda *a, **k: (calls.append(1), real_multinomial(*a, **k))[1]
    import unittest.mock
    with unittest.mock.patch.object(torch, "multinomial", with_patch):
        with torch.no_grad():
            out1 = model.generate(idx, max_new_tokens=5, greedy=True)
            out2 = model.generate(idx, max_new_tokens=5, greedy=True)

    assert len(calls) == 0
    # Deterministic — greedy on the same input always produces the same output.
    assert out1.tolist() == out2.tolist()


async def _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id: int) -> int:
    """Shared setup for Phase 2/3/4 tests — isolated data_dir, config, checkpoint,
    PAUSED status, and a REAL training_runs row (via create_training_run, not a
    hardcoded id + no-op UPDATE — Phase 4's diagnostic_sessions table has a real
    FOREIGN KEY on run_id, which exposed that the old hardcoded-id pattern never
    actually inserted a row; UPDATE on a nonexistent id just silently matches
    zero rows, which every non-FK-constrained test happened not to notice).
    Returns the real run_id."""
    import torch
    from backend.training import artifacts
    from backend.training.templates import TEMPLATE_REGISTRY

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG))
    model = TEMPLATE_REGISTRY["transformer"]["build_model"](TRANSFORMER_CONFIG)
    torch.save({"model_state": model.state_dict(), "config": TRANSFORMER_CONFIG}, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")
    return run_id


async def test_attention_capture_returns_causal_weights(temp_db, client, monkeypatch, tmp_path):
    """Requesting attention_layer/head returns row-normalized, causal-masked weights."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={"attention_layer": 0, "attention_head": 0},
    )
    assert resp.status_code == 200
    snapshot = resp.json()

    attn = snapshot["attention"]
    assert attn["available"] is True
    assert attn["layer"] == 0
    assert attn["head"] == 0
    weights = attn["weights"]
    n = len(weights)
    assert n == len(attn["token_labels"])
    for i, row in enumerate(weights):
        # Causal: position i can only attend to positions <= i
        for j in range(i + 1, n):
            assert row[j] == pytest.approx(0.0, abs=1e-6)
        assert sum(row) == pytest.approx(1.0, abs=1e-4)


async def test_attention_not_computed_when_not_requested(temp_db, client, monkeypatch, tmp_path):
    """Omitting attention_layer/head keeps attention unavailable with 'Not requested'."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})
    assert resp.status_code == 200
    snapshot = resp.json()
    assert snapshot["attention"] == {"available": False, "reason": "Not requested"}


async def test_generate_streams_tokens_then_final_snapshot(temp_db, client, monkeypatch, tmp_path):
    """POST /generate streams N token events then one done event with a full snapshot."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    async with client.stream(
        "POST", f"/api/training/{run_id}/diagnostics/{session_id}/generate",
        json={"max_new_tokens": 3},
    ) as resp:
        assert resp.status_code == 200
        body = ""
        async for chunk in resp.aiter_text():
            body += chunk

    frames = [f for f in body.split("\n\n") if f.strip()]
    events = []
    for frame in frames:
        lines = frame.strip().split("\n")
        event_type = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event_type, data))

    token_events = [e for e in events if e[0] == "token"]
    done_events = [e for e in events if e[0] == "done"]
    assert len(token_events) == 3
    assert len(done_events) == 1
    # generation_step increases monotonically across the 3 token events
    steps = [e[1]["generation_step"] for e in token_events]
    assert steps == sorted(steps)
    final_snapshot = done_events[0][1]["final_snapshot"]
    assert final_snapshot["schema_version"] == 1
    assert "lm_head" in final_snapshot


async def test_qkv_detail_returns_vectors_when_requested(temp_db, client, monkeypatch, tmp_path):
    """qkv_detail=True returns one Q/K/V vector per position (not just the
    last token) — the frontend's position stepper needs one per position."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={"attention_layer": 0, "attention_head": 0, "qkv_detail": True},
    )
    assert resp.status_code == 200
    attn = resp.json()["attention"]
    assert attn["available"] is True
    qkv = attn["qkv_detail"]
    num_positions = len(qkv["positions"])
    # +1: qkv_detail is computed after this step's new token is appended to
    # token_history (prompt is under the 12-position cap either way).
    assert num_positions == len("The king") + 1
    assert qkv["positions"] == list(range(num_positions))
    assert len(qkv["tokens"]) == num_positions
    assert len(qkv["q"]) == num_positions
    assert len(qkv["k"]) == num_positions
    assert len(qkv["v"]) == num_positions
    assert isinstance(qkv["q"][0], list) and len(qkv["q"][0]) > 0


async def test_qkv_detail_capped_to_last_12_positions(temp_db, client, monkeypatch, tmp_path):
    """A prompt longer than DIAGNOSTIC_POSITION_WINDOW (12) must only return
    the most recent 12 positions' Q/K/V and top-k, not all of them —
    payload-size cap confirmed live, not just a config value that's never
    exercised. qkv_detail's window is one position further than
    top_k_by_position's (see the comment in diagnostics.py's
    _execute_forward_pass): qkv is computed after this step's new token is
    appended to token_history, top_k_by_position before."""
    from backend.training.diagnostics import DIAGNOSTIC_POSITION_WINDOW

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    long_prompt = "The king said many words indeed"
    assert len(long_prompt) > DIAGNOSTIC_POSITION_WINDOW
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": long_prompt, "top_k": 5, "max_prompt_tokens": len(long_prompt)},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={"attention_layer": 0, "attention_head": 0, "qkv_detail": True},
    )
    assert resp.status_code == 200
    body = resp.json()

    pre_append_len = len(long_prompt)  # sequence length used for top_k_by_position
    post_append_len = pre_append_len + 1  # +1 for the token generated this step, used for qkv_detail

    qkv = body["attention"]["qkv_detail"]
    assert len(qkv["positions"]) == DIAGNOSTIC_POSITION_WINDOW
    assert qkv["positions"] == list(range(post_append_len - DIAGNOSTIC_POSITION_WINDOW, post_append_len))

    top_k_by_position = body["lm_head"]["top_k_by_position"]
    assert len(top_k_by_position) == DIAGNOSTIC_POSITION_WINDOW
    assert top_k_by_position[-1]["position"] == pre_append_len - 1
    assert top_k_by_position[0]["position"] == pre_append_len - DIAGNOSTIC_POSITION_WINDOW
    assert len(top_k_by_position[0]["top_k"]) == 5


async def test_qkv_detail_omitted_when_not_requested(temp_db, client, monkeypatch, tmp_path):
    """Phase 4: qkv_detail defaults false — no qkv_detail key in the response."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={"attention_layer": 0, "attention_head": 0},
    )
    assert resp.status_code == 200
    attn = resp.json()["attention"]
    assert attn["available"] is True
    assert "qkv_detail" not in attn


async def test_diagnostic_session_persisted_after_generate(temp_db, client, monkeypatch, tmp_path):
    """Phase 4: /generate's done event writes one diagnostic_sessions row."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    async with client.stream(
        "POST", f"/api/training/{run_id}/diagnostics/{session_id}/generate",
        json={"max_new_tokens": 2},
    ) as resp:
        body = ""
        async for chunk in resp.aiter_text():
            body += chunk
    assert "event: done" in body

    conn = await db.get_db()
    cursor = await conn.execute("SELECT COUNT(*) FROM diagnostic_sessions WHERE run_id = ?", (run_id,))
    (count,) = await cursor.fetchone()
    await conn.close()
    assert count == 1


async def test_get_diagnostic_snapshot_for_run_tracks_run_to_session(temp_db, client, monkeypatch, tmp_path):
    """Chatbot grounding accessor: after starting a session and stepping,
    get_diagnostic_snapshot_for_run(run_id) finds it without a session_id."""
    from backend.api.training import get_diagnostic_snapshot_for_run

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})
    stepped = resp.json()

    snapshot = await get_diagnostic_snapshot_for_run(run_id)
    assert snapshot is not None
    assert snapshot["diagnostic_session_id"] == session_id
    assert snapshot["generation_step"] == stepped["generation_step"]


async def test_diagnostics_start_reads_temperature_from_config(temp_db, client, monkeypatch, tmp_path):
    """> / >> must sample with the same temperature the Generate button uses
    (config.inference.temperature) — not a hardcoded/greedy default. See
    docs/DESIGN_DECISIONS.md."""
    from backend.training import artifacts, diagnostics
    from backend.training.templates import TEMPLATE_REGISTRY
    import torch

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    config = {**TRANSFORMER_CONFIG, "inference": {"temperature": 1.5}}
    (rd / "config.json").write_text(json.dumps(config))
    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    torch.save({"model_state": model.state_dict(), "config": config}, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    session = diagnostics.get_session(session_id)
    assert session.temperature == 1.5


async def test_diagnostics_start_defaults_temperature_when_config_omits_it(temp_db, client, monkeypatch, tmp_path):
    """TRANSFORMER_CONFIG (shared fixture) has no "inference" key — must fall
    back to the same 0.8 default model.generate() uses, not error."""
    from backend.training import diagnostics

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    assert diagnostics.get_session(session_id).temperature == 0.8


async def test_diagnostics_start_reads_decoding_mode_from_config(temp_db, client, monkeypatch, tmp_path):
    """decoding_mode is a new config.inference setting (2026-07-15), same
    source Generate uses via prompt_paused_model — > / >> must pick it up
    too. See docs/DESIGN_DECISIONS.md."""
    from backend.training import artifacts, diagnostics
    from backend.training.templates import TEMPLATE_REGISTRY
    import torch

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    config = {**TRANSFORMER_CONFIG, "inference": {"decoding_mode": "greedy"}}
    (rd / "config.json").write_text(json.dumps(config))
    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    torch.save({"model_state": model.state_dict(), "config": config}, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    assert diagnostics.get_session(session_id).decoding_mode == "greedy"


async def test_diagnostics_start_defaults_decoding_mode_to_sample(temp_db, client, monkeypatch, tmp_path):
    from backend.training import diagnostics

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    assert diagnostics.get_session(session_id).decoding_mode == "sample"


async def test_diagnostic_step_uses_argmax_in_greedy_mode(temp_db, client, monkeypatch, tmp_path):
    """When decoding_mode=greedy, step-through must select via torch.argmax
    and never call torch.multinomial — the selected token must always be
    the top-k list's #1 entry (deterministic). See docs/DESIGN_DECISIONS.md."""
    from backend.training import artifacts, diagnostics
    from backend.training.templates import TEMPLATE_REGISTRY
    import torch

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    config = {**TRANSFORMER_CONFIG, "inference": {"decoding_mode": "greedy"}}
    (rd / "config.json").write_text(json.dumps(config))
    model = TEMPLATE_REGISTRY["transformer"]["build_model"](config)
    torch.save({"model_state": model.state_dict(), "config": config}, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    multinomial_calls = []
    real_multinomial = torch.multinomial
    monkeypatch.setattr(torch, "multinomial", lambda *a, **k: (multinomial_calls.append(1), real_multinomial(*a, **k))[1])

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})
    assert resp.status_code == 200
    body = resp.json()

    assert len(multinomial_calls) == 0
    assert body["generated_token"]["id"] == body["lm_head"]["top_k"][0]["token_id"]


async def test_diagnostic_step_samples_instead_of_greedy(temp_db, client, monkeypatch, tmp_path):
    """Step-through must call torch.multinomial (real sampling), not
    torch.argmax/topk[0] (greedy) — regression test for the >  / >> vs
    Generate behavior mismatch. See docs/DESIGN_DECISIONS.md."""
    import torch
    from backend.training import diagnostics as diag_module

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    real_multinomial = torch.multinomial
    calls = []

    def spy_multinomial(probs, *args, **kwargs):
        calls.append(probs)
        return real_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(diag_module.torch, "multinomial", spy_multinomial)

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})

    assert resp.status_code == 200
    assert len(calls) == 1


async def test_generate_samples_instead_of_greedy(temp_db, client, monkeypatch, tmp_path):
    """/generate (>>) must also sample with torch.multinomial per token, not
    torch.argmax — its token-selection loop lives separately from
    _execute_forward_pass's, in backend/api/training.py. See
    docs/DESIGN_DECISIONS.md."""
    import torch

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    real_multinomial = torch.multinomial
    calls = []

    def spy_multinomial(probs, *args, **kwargs):
        calls.append(probs)
        return real_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", spy_multinomial)

    async with client.stream(
        "POST", f"/api/training/{run_id}/diagnostics/{session_id}/generate",
        json={"max_new_tokens": 3},
    ) as resp:
        assert resp.status_code == 200
        async for _ in resp.aiter_text():
            pass

    assert len(calls) == 3


async def test_peek_recomputes_attention_for_a_new_head_without_advancing(temp_db, client, monkeypatch, tmp_path):
    """/peek lets the UI refresh attention for a newly-selected head without
    a full step — must NOT sample a new token or advance generation_step.
    Real bug report, 2026-07-14: changing Head in Inspector did nothing
    until the next > click. See docs/DESIGN_DECISIONS.md."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    # A real step, head 0 — establishes generation_step=1 and a token count baseline.
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/step",
        json={"attention_layer": 0, "attention_head": 0},
    )
    stepped = resp.json()
    assert stepped["attention"]["head"] == 0
    assert stepped["generation_step"] == 1
    token_count_after_step = stepped["generated_token"]["position"] + 1

    # Peek at a different head — must return that head's real weights,
    # without sampling a new token or advancing generation_step.
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/peek",
        json={"attention_layer": 0, "attention_head": 1},
    )
    assert resp.status_code == 200
    peeked = resp.json()
    assert peeked["attention"]["available"] is True
    assert peeked["attention"]["head"] == 1
    assert peeked["generation_step"] == 1  # unchanged — peek doesn't advance
    assert peeked["generated_token"]["position"] + 1 == token_count_after_step  # no new token appended

    # A second peek, back to head 0, still doesn't advance — confirms peek
    # is idempotent/non-mutating, not just "didn't advance once."
    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/peek",
        json={"attention_layer": 0, "attention_head": 0},
    )
    peeked_again = resp.json()
    assert peeked_again["generation_step"] == 1
    assert peeked_again["generated_token"]["position"] + 1 == token_count_after_step


async def test_peek_before_any_step_still_works(temp_db, client, monkeypatch, tmp_path):
    """Peek must work even if the user changes Head before ever clicking >
    — a fresh session has no prior step, only the prompt."""
    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/{session_id}/peek",
        json={"attention_layer": 0, "attention_head": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["attention"]["available"] is True
    assert body["generation_step"] == 0


async def test_generic_node_captures_position_vectors(temp_db, client, monkeypatch, tmp_path):
    """Every node (not just attention/lm_head) now retains raw per-position
    vectors, capped to DIAGNOSTIC_POSITION_WINDOW, so the Inspector can show
    real values instead of just shape+summary stats. See
    docs/DESIGN_DECISIONS.md."""
    from backend.training.diagnostics import DIAGNOSTIC_POSITION_WINDOW

    exp_id = temp_db
    run_id = await _setup_paused_run_with_checkpoint(monkeypatch, tmp_path, exp_id)

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king said", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]

    for node_id in ["embedding", "block.0.ln1", "block.0.attention", "block.0.ln2", "block.0.mlp", "final_norm"]:
        pv = nodes[node_id]["position_vectors"]
        assert pv is not None, f"{node_id} missing position_vectors"
        assert len(pv["positions"]) <= DIAGNOSTIC_POSITION_WINDOW
        assert len(pv["vectors"]) == len(pv["positions"])
        assert len(pv["vectors"][0]) == TRANSFORMER_CONFIG["model"]["n_embd"]


async def test_moe_template_captures_all_nodes_not_just_moe_block(temp_db, client, monkeypatch, tmp_path):
    """Regression test: the MoE template's diagnostic hooks only ever
    branched on isinstance(output, tuple) — true only for the .moe node
    itself (which returns (x, drop_rate)). Every other node (embedding,
    ln1, attention, ln2, final_norm, lm_head) returns a plain tensor and
    fell into the else branch, silently getting empty shape/summary/no
    vectors. Found live while adding position_vectors capture, 2026-07-14.
    See docs/DESIGN_DECISIONS.md."""
    import torch
    from backend.training import artifacts
    from backend.training.templates import TEMPLATE_REGISTRY

    moe_config = {
        "template": "moe",
        "device": "cpu",
        "model": {
            "vocab_size": 65, "block_size": 128, "n_embd": 192, "n_head": 6,
            "n_layer": 2, "dropout": 0.1, "pos_encoding": "rope", "activation": "gelu",
            "num_experts": 8, "top_k": 2, "capacity_factor": 1.25,
        },
        "training": {
            "batch_size": 64, "learning_rate": 3e-4, "max_iters": 1000,
            "eval_interval": 20, "eval_iters": 2, "optimizer": "adamw",
        },
    }
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    exp_id = temp_db
    run_id = await db.create_training_run(exp_id, device="cpu", execution_backend="local")
    rd = artifacts.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(moe_config))
    model = TEMPLATE_REGISTRY["moe"]["build_model"](moe_config)
    torch.save({"model_state": model.state_dict(), "config": moe_config}, artifacts.checkpoint_path(run_id))
    artifacts.write_status(run_id, {"status": RunStatus.PAUSED, "current_step": 10, "total_steps": 100})
    await db.update_training_run(run_id, status=RunStatus.PAUSED, device="cpu", execution_backend="local")

    resp = await client.post(
        f"/api/training/{run_id}/diagnostics/start",
        json={"prompt": "The king", "top_k": 5, "max_prompt_tokens": 32},
    )
    session_id = resp.json()["diagnostic_session_id"]

    resp = await client.post(f"/api/training/{run_id}/diagnostics/{session_id}/step", json={})
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]

    # Before the fix, all of these had output_shape=[] and empty summary.
    for node_id in ["embedding", "block.0.ln1", "block.0.attention", "block.0.ln2", "final_norm"]:
        assert nodes[node_id]["output_shape"] != [], f"{node_id} has empty output_shape (pre-fix bug)"
        assert nodes[node_id]["summary"] != {}, f"{node_id} has empty summary (pre-fix bug)"
        assert nodes[node_id]["position_vectors"] is not None

    # The .moe node itself always worked (its output genuinely is a tuple).
    assert nodes["block.0.moe"]["output_shape"] != []
