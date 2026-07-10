import pytest

from backend.chatbot import client as tf_client
from config.settings import settings


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [_FakeChoice(content)] if content is not None else []
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(chunks)


def test_is_configured_false_without_key(monkeypatch):
    monkeypatch.setattr(settings, "nebius_key", None)
    assert tf_client.is_configured() is False


def test_is_configured_true_with_key(monkeypatch):
    monkeypatch.setattr(settings, "nebius_key", "fake-key-value")
    assert tf_client.is_configured() is True


async def test_stream_completion_yields_deltas_then_usage(monkeypatch):
    chunks = [
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" world"),
        _FakeChunk(content=None, usage=_FakeUsage(10, 5, 15)),
    ]
    monkeypatch.setattr(tf_client, "_get_client", lambda: _FakeClient(chunks))

    results = [
        (delta, usage)
        async for delta, usage in tf_client.stream_completion([{"role": "user", "content": "hi"}])
    ]

    deltas = [d for d, _ in results if d]
    assert "".join(deltas) == "Hello world"
    usages = [u for _, u in results if u is not None]
    assert usages == [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}]
