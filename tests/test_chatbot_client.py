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
    def __init__(self, chunks, nonstream_response=None):
        self._chunks = chunks
        self._nonstream_response = nonstream_response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream") is False:
            return self._nonstream_response
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks, nonstream_response=None):
        self.completions = _FakeCompletions(chunks, nonstream_response)


class _FakeClient:
    def __init__(self, chunks, nonstream_response=None):
        self.chat = _FakeChat(chunks, nonstream_response)


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


class _FakeNonStreamChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeNonStreamResponse:
    def __init__(self, content, usage):
        self.choices = [_FakeNonStreamChoice(content)]
        self.usage = usage


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


async def test_stream_completion_with_tool_context_but_no_lookup_hint_streams_normally(monkeypatch):
    """No digits/keywords in the message -> skip the non-streaming preflight entirely."""
    chunks = [
        _FakeChunk(content="Sure, "),
        _FakeChunk(content="happy to help."),
        _FakeChunk(content=None, usage=_FakeUsage(3, 4, 7)),
    ]
    fake_client = _FakeClient(chunks)
    monkeypatch.setattr(tf_client, "_get_client", lambda: fake_client)

    results = [
        (delta, usage)
        async for delta, usage in tf_client.stream_completion(
            [{"role": "user", "content": "hello there, how are you?"}],
            tool_context={"run_ids": ["run-1"], "template": "transformer"},
        )
    ]

    deltas = [d for d, _ in results if d]
    assert "".join(deltas) == "Sure, happy to help."
    # Only the streaming call should have happened - no preflight, no tools= sent.
    assert len(fake_client.chat.completions.calls) == 1
    assert fake_client.chat.completions.calls[0]["stream"] is True
    assert "tools" not in fake_client.chat.completions.calls[0]


async def test_stream_completion_with_lookup_hint_uses_nonstream_preflight_no_tool_calls(monkeypatch):
    """Digits/keywords present -> non-streaming preflight runs; no tool_calls -> answer yielded directly."""
    nonstream_response = _FakeNonStreamResponse(
        "The loss at step 100 was 0.42.", _FakeUsage(20, 10, 30)
    )
    fake_client = _FakeClient(chunks=[], nonstream_response=nonstream_response)
    monkeypatch.setattr(tf_client, "_get_client", lambda: fake_client)

    results = [
        (delta, usage)
        async for delta, usage in tf_client.stream_completion(
            [{"role": "user", "content": "what was the loss at step 100?"}],
            tool_context={"run_ids": ["run-1"], "template": "transformer"},
        )
    ]

    deltas = [d for d, _ in results if d]
    assert "".join(deltas) == "The loss at step 100 was 0.42."
    usages = [u for _, u in results if u is not None]
    assert usages == [{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}]
    # Only the non-streaming preflight call should have happened - no fallback stream call.
    assert len(fake_client.chat.completions.calls) == 1
    assert fake_client.chat.completions.calls[0]["stream"] is False
    assert fake_client.chat.completions.calls[0]["tools"] == tf_client.TOOL_SCHEMAS


def test_looks_like_lookup_needed_true_for_digits_and_keywords():
    assert tf_client._looks_like_lookup_needed("what happened at step 100") is True
    assert tf_client._looks_like_lookup_needed("what's the learning rate?") is True


def test_looks_like_lookup_needed_false_for_plain_chat():
    assert tf_client._looks_like_lookup_needed("hello, how are you today?") is False
