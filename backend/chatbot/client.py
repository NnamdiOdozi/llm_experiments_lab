"""Thin wrapper around the Nebius Token Factory OpenAI-compatible API."""

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from backend.chatbot.tools import TOOL_SCHEMAS, execute_tool_call
from backend.logging_config import chatbot_log, error_log
from config.settings import settings

_client: AsyncOpenAI | None = None


def is_configured() -> bool:
    return bool(settings.nebius_key)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.token_factory_base_url,
            api_key=settings.nebius_key,
        )
    return _client


async def stream_completion(
    messages: list[dict], tool_context: dict | None = None
) -> AsyncIterator[tuple[str, dict | None]]:
    """Stream a chat completion from Token Factory.

    Yields (text_delta, usage) pairs. usage is None on every chunk except
    the final one, per the OpenAI streaming spec with include_usage set.
    """
    client = _get_client()
    start = time.perf_counter()
    chatbot_log.info(
        "Token Factory request: model=%s messages=%d", settings.token_factory_model, len(messages)
    )
    try:
        stream_messages = list(messages)
        if tool_context is not None:
            first = await client.chat.completions.create(
                model=settings.token_factory_model,
                messages=stream_messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=False,
            )
            first_choice = first.choices[0]
            first_message = first_choice.message
            tool_calls = getattr(first_message, "tool_calls", None) or []
            if tool_calls:
                if hasattr(first_message, "model_dump"):
                    stream_messages.append(first_message.model_dump(exclude_none=True))
                else:
                    stream_messages.append(first_message)
                for call in tool_calls:
                    result = execute_tool_call(
                        call.function.name,
                        call.function.arguments,
                        allowed_run_ids=tool_context.get("run_ids", []),
                        template=tool_context.get("template", "transformer"),
                    )
                    stream_messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result}
                    )
                chatbot_log.info("Executed %d chatbot tool call(s)", len(tool_calls))
            else:
                usage = getattr(first, "usage", None)
                content = first_message.content or ""
                if content:
                    yield content, None
                if usage is not None:
                    yield "", {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                return

        stream = await client.chat.completions.create(
            model=settings.token_factory_model,
            messages=stream_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = None
            if chunk.usage is not None:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            delta = ""
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
            if delta or usage is not None:
                yield delta, usage
        elapsed_ms = (time.perf_counter() - start) * 1000
        chatbot_log.info("Token Factory request complete: %.0fms", elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        error_log.error("Token Factory request failed after %.0fms: %s", elapsed_ms, exc)
        raise
