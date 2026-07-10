import json

import pytest
from httpx import ASGITransport, AsyncClient

from backend import db
from backend.chatbot import client as tf_client
from backend.main import app


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment(
        "Test experiment", {"template": "transformer", "description": "baseline"}
    )
    return exp_id


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_get_messages_empty_for_new_experiment(temp_db, client):
    resp = await client.get(f"/api/chatbot/{temp_db}/messages")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_messages_404_for_unknown_experiment(temp_db, client):
    resp = await client.get("/api/chatbot/999999/messages")
    assert resp.status_code == 404


async def test_post_message_returns_503_when_not_configured(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: False)
    resp = await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "hello"})
    assert resp.status_code == 503


async def test_post_message_streams_and_persists(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: True)

    async def fake_stream(messages):
        yield "Hello", None
        yield " there", None
        yield "", {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52}

    monkeypatch.setattr(tf_client, "stream_completion", fake_stream)

    resp = await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "hi"})

    assert resp.status_code == 200
    assert "data:" in resp.text
    assert "event: done" in resp.text

    messages = await db.get_chat_messages(temp_db)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello there"
    assert messages[1]["total_tokens"] == 52


async def test_post_message_history_excludes_message_being_sent(temp_db, client, monkeypatch):
    monkeypatch.setattr(tf_client, "is_configured", lambda: True)
    captured = {}

    async def fake_stream(messages):
        captured["messages"] = messages
        yield "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(tf_client, "stream_completion", fake_stream)

    await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "first"})
    await client.post(f"/api/chatbot/{temp_db}/message", json={"message": "second"})

    last_user_turn = captured["messages"][-1]["content"]
    assert "second" in last_user_turn
    occurrences = sum(1 for m in captured["messages"] if m["content"] == "first")
    assert occurrences == 1
