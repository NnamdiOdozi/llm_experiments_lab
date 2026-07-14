"""Grounded chatbot endpoints — SSE chat streaming and history."""

import inspect
import json
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import db
from backend.chatbot import client as tf_client
from backend.chatbot.context import assemble_messages, get_tool_context
from backend.logging_config import chatbot_log
from config.settings import settings

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


class ChatMessageRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    # None clears feedback (e.g. clicking an already-selected thumb again)
    feedback: Literal["up", "down"] | None = None


@router.get("/{experiment_id}/messages")
async def get_messages(experiment_id: int):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return await db.get_chat_messages(experiment_id)


@router.delete("/{experiment_id}/messages")
async def clear_messages(experiment_id: int):
    """Clears all chat history for an experiment — lets the user reset a
    stuck/confused Lab Assistant conversation without starting a new
    experiment (which would also lose all its runs). Direct user request,
    2026-07-14. See docs/DESIGN_DECISIONS.md."""
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    deleted = await db.clear_chat_messages(experiment_id)
    chatbot_log.info("Chat history cleared: experiment_id=%d deleted=%d", experiment_id, deleted)
    return {"experiment_id": experiment_id, "deleted": deleted}


@router.patch("/messages/{message_id}/feedback")
async def set_feedback(message_id: int, req: FeedbackRequest):
    updated = await db.set_chat_message_feedback(message_id, req.feedback)
    if not updated:
        raise HTTPException(404, "Message not found")
    chatbot_log.info("Feedback set: message_id=%d feedback=%s", message_id, req.feedback)
    return {"ok": True}


@router.post("/{experiment_id}/message")
async def post_message(experiment_id: int, req: ChatMessageRequest):
    exp = await db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")

    if not tf_client.is_configured():
        chatbot_log.warning(
            "Chat request for experiment_id=%d but NEBIUS_KEY not configured", experiment_id
        )
        raise HTTPException(503, "Chatbot unavailable — no Token Factory API key configured")

    # Fetch history BEFORE writing the new user message, so assemble_messages
    # doesn't see the current message twice (once in history, once stapled
    # as the current turn).
    history = await db.get_chat_messages(
        experiment_id, limit=settings.chatbot_history_window_turns * 2
    )
    await db.add_chat_message(experiment_id, "user", req.message)

    runs = await db.list_runs_for_experiment(experiment_id)
    latest_run = runs[0] if runs else None
    messages = assemble_messages(exp, latest_run, history, req.message)
    tool_context = get_tool_context(exp, runs)

    async def event_stream():
        full_text = []
        usage_info = None
        start = time.perf_counter()
        try:
            stream_kwargs = {}
            if "tool_context" in inspect.signature(tf_client.stream_completion).parameters:
                stream_kwargs["tool_context"] = tool_context
            async for delta, usage in tf_client.stream_completion(messages, **stream_kwargs):
                if delta:
                    full_text.append(delta)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
                if usage is not None:
                    usage_info = usage
        except Exception as exc:
            chatbot_log.error("Chat stream failed for experiment_id=%d: %s", experiment_id, exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        latency_ms = int((time.perf_counter() - start) * 1000)
        assistant_text = "".join(full_text)
        message_id = await db.add_chat_message(
            experiment_id,
            "assistant",
            assistant_text,
            prompt_tokens=(usage_info or {}).get("prompt_tokens"),
            completion_tokens=(usage_info or {}).get("completion_tokens"),
            total_tokens=(usage_info or {}).get("total_tokens"),
            latency_ms=latency_ms,
        )
        # The client only has a local placeholder id for this message until
        # now (assigned before the real DB row existed) — send the real one
        # back so feedback (thumbs up/down) can PATCH the actual row instead
        # of 404ing against an id that was never persisted.
        yield f"event: done\ndata: {json.dumps({'usage': usage_info, 'message_id': message_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
