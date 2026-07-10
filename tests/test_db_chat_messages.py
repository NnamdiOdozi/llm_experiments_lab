import pytest

from backend import db


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lab.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    await db.init_db()
    exp_id = await db.create_experiment("Test experiment", {"template": "transformer"})
    return exp_id


async def test_add_and_get_chat_messages(temp_db):
    exp_id = temp_db
    await db.add_chat_message(exp_id, "user", "What does this mean?")
    await db.add_chat_message(
        exp_id, "assistant", "It means the loss is decreasing.",
        prompt_tokens=100, completion_tokens=20, total_tokens=120, latency_ms=850,
    )

    messages = await db.get_chat_messages(exp_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What does this mean?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["prompt_tokens"] == 100
    assert messages[1]["total_tokens"] == 120


async def test_get_chat_messages_respects_limit_and_order(temp_db):
    exp_id = temp_db
    for i in range(5):
        await db.add_chat_message(exp_id, "user", f"message {i}")

    messages = await db.get_chat_messages(exp_id, limit=2)

    assert len(messages) == 2
    # Must be the two most recent, in chronological order
    assert messages[0]["content"] == "message 3"
    assert messages[1]["content"] == "message 4"


async def test_get_chat_messages_empty_for_unknown_experiment(temp_db):
    messages = await db.get_chat_messages(999999)
    assert messages == []
