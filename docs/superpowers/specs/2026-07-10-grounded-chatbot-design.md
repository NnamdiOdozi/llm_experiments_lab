# Grounded Chatbot — Design Spec

**Date:** 2026-07-10
**Status:** Approved — ready for implementation plan

## 1. What this is

A lab-demonstrator chatbot (per project doc §9) grounded in the user's current
experiment: config, live loss curve, last change made, and the source code of
the architecture they're actually running. It is **not** a generic chat
assistant and, in v1, it is **read-only/advisory** — it cannot edit code,
generate configs, or trigger runs. Code remains static, editable only via the
existing side-panel Config UI. Chatbot-driven config/run generation is an
explicit Phase 2 item, not built now.

This spec targets the **actual current codebase** (SQLite via `aiosqlite`,
in-process CPU/GPU training, no auth, no S3, no Postgres) — not the
aspirational architecture described elsewhere in the project discussion doc
(§11), which describes a later production target.

## 2. Retrieval strategy: eager injection, not RAG or tool-calling

All grounding data for a given experiment — config JSON, ~20 recent loss
points, one audit-log line, one preset description, the relevant template's
source (`model.py`/`data.py`, ~104–225 lines) — totals roughly 3–6K tokens.
Against Qwen3-235B-A22B-Thinking's 262K-token context window, this is small
enough to load in full on every turn. There is no search problem to solve:
`config_json["template"]` deterministically identifies which source file
pair is relevant, so there's nothing for the model to search for.

**Explicitly out of scope for v1:** agentic retrieval / tool-calling for
arbitrary codebase Q&A (e.g. "how does `runner.py`'s pause/resume work").
That part of the codebase is unbounded and would need a read-tool with a
hard security allow-list (never `.env`, never DB connection internals) —
real added complexity, deferred to Phase 2. For v1, the system prompt
instructs the model to say "I don't have visibility into that part of the
implementation" rather than fabricate an answer, when a question falls
outside the injected context. General ML/theory questions ("what does RoPE
mean") need no retrieval at all — the base model already knows this from
pretraining.

## 3. Context tiers and message ordering

Ordered static-first, volatile-last. Two motivations, not one:

- **Quality / recency**: the most relevant-to-this-question facts sit next
  to the user's actual message, not buried mid-prompt.
- **Cache-readiness, not cache savings**: Nebius Token Factory does **not**
  currently bill discounted cached tokens (confirmed — open feature request
  at [ideas.nebius.com](https://ideas.nebius.com/p/support-impliciteexplicite-prompt-caching),
  not shipped; see [API reference](https://docs.tokenfactory.nebius.com/api-reference/introduction)).
  Ordering this way costs nothing, is forward-compatible if Nebius ships
  billed caching, and may help engine-level TTFT even without a billing
  discount (unconfirmed, plausible, not load-bearing for this design).

| Tier | Content | Volatility | Where in the message list |
|---|---|---|---|
| 1 | Chatbot persona, behavior rules, UI-affordances description (config panel vs. layer stack) | Fully static, app-wide | `system` message |
| 2 | Current template's `model.py`+`data.py` source, preset description | Static per session (changes only if user switches experiments) | Early context message, right after `system` |
| 3 | Prior conversation turns (sliding window, see §5) | Append-only | Between tier 2 and the current turn |
| 4 | Current step, recent loss trend (last ~20 points), last audit-logged change, log tail | Changes almost every turn | Stapled to the **current** user message, not the system prompt |

## 4. What "last change" and "log tail" actually read

- **Last change**: every `audit_log.info(...)` call in `backend/api/experiments.py`
  embeds the experiment's numeric ID as either `id=%d` or `experiment_id=%d`
  in the message text (both forms exist — lines 40, 50, 84). The context
  builder tails the current session's log file (reusing
  `logging_config.get_log_path()` — no path re-derivation), filters for
  `lab.audit` lines matching either ID pattern, takes the most recent match.
- **Log tail**: last `settings.chatbot_log_tail_lines` (default 50) lines of
  the same session log file, any category — gives visibility into errors,
  pause/resume events, not just audit entries.
- Known limitation, accepted for a POC: this only searches the *current
  server session's* log file. No cross-session search. Adequate at current
  scale (one file per session); would need revisiting if session log files
  grow very large or cross-session history becomes important.

## 5. Conversation history: sliding window, not summarization

Last 10 turns (20 messages: 10 user + 10 assistant) sent to the model as
tier 3. Full history still persisted to DB and shown in the UI regardless —
the window only affects what's sent to the LLM, not what's stored or
displayed. Rejected summarization (extra LLM call, new failure mode, not
worth it for a chat that's fundamentally "explain *this* run right now" not
a long-running relationship) and token-counted truncation (adds a
dependency for a payload this small). Window size configurable via
`settings.chatbot_history_window_turns`.

## 6. Data model

New table in `backend/db.py`:

```sql
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    role TEXT NOT NULL,              -- 'user' | 'assistant'
    content TEXT NOT NULL,
    prompt_tokens INTEGER,           -- null for user-role rows
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Token-usage columns exist for observability (project convention: log
metrics generously, queryable after the fact) — lets you later query "is
prompt size creeping up over a long session" without guessing. Populated
from the OpenAI-compatible `usage` object, requested via
`stream_options={"include_usage": True}`.

One function in `db.py`, not two: `get_chat_messages(experiment_id,
limit=None)` — a single optional `limit` param serves both "full history for
the UI" and "last N for the sliding window," rather than two near-duplicate
functions.

## 7. API contract

- `POST /api/chatbot/{experiment_id}/message` — body `{message: str}`.
  Writes the user message to `chat_messages`, streams the assistant reply as
  Server-Sent Events (`text/event-stream`). Accumulates the full text
  server-side; writes one `chat_messages` row (with usage stats) once the
  stream ends. If `chatbot.client.is_configured()` is false (no
  `NEBIUS_KEY`), returns immediately with a clear "chatbot unavailable"
  response — no stream attempted.
- `GET /api/chatbot/{experiment_id}/messages` — full history, for the panel
  to hydrate on page load.

## 8. Config and secrets

`config/settings.py` additions (no hardcoded values in source):

```python
token_factory_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
token_factory_model: str = "Qwen/Qwen3-235B-A22B-Thinking-2507"
chatbot_log_tail_lines: int = 50
chatbot_history_window_turns: int = 10
```

Secret: reuses the existing `NEBIUS_KEY` from `.env` (confirmed to be the
Token Factory / AI Studio inference key, distinct from the IAM
service-account `.pem` used by the `nebius` CLI for compute). No new secret
variable introduced. `OPENAI_API_KEY` in `.env` is confirmed unrelated (a
real OpenAI key for something else) — not read or referenced anywhere in
this feature.

One deliberate exception to "no hardcoding": the system-prompt text itself
(tier 1, ~20 lines) is a Python string constant in `context.py`, not
externalized to `.toml`/`.env`. Rationale: it's prose defining chatbot
behavior — closer to an error message or docstring than a
deployment-dependent configurable value. Revisit if there's ever a need to
edit persona/behavior without a code change.

## 9. Error handling

- `NEBIUS_KEY` missing at startup → `session_log.warning(...)`, chat panel
  shows "Chatbot unavailable — no API key configured" instead of the app
  crashing.
- Token Factory request fails or times out mid-stream → SSE emits an
  `event: error` frame; frontend shows an inline error bubble; no partial
  garbage assistant message is written to `chat_messages`.
- All Token Factory calls logged via a new `chatbot_log` category in
  `logging_config.py` (request sent, latency, usage if available, errors) —
  consistent with existing `[REQUEST]/[TRAINING]/[AUDIT]/[SESSION]`
  categories.

## 10. Rejected approaches (for the record)

- **Frontend calls Token Factory directly** — would expose `NEBIUS_KEY` in
  browser-shipped code. Rejected: real security regression, inconsistent
  with the rest of the app (backend mediates all external calls).
- **Separate chatbot microservice** — unnecessary process/deployment
  complexity for a single FastAPI monolith at this scale. Revisit only if
  the whole app decomposes (e.g. training genuinely moves to Nebius
  serverless jobs).
- **Agentic tool-calling for grounding** — rejected for the *live-state*
  context (tier 1–2 data) because it's small and known in advance; retained
  as a Phase 2 idea specifically for *unbounded* codebase Q&A (§2).

## 11. File budget

Calibrated against existing file sizes in this codebase (`db.py` ≈10–15
lines/function, `useApi.ts` 80 lines for the whole app's REST calls,
`TrainingControls.tsx` 154 lines for a materially more complex component).

| File | Lines | Contents |
|---|---|---|
| `backend/chatbot/__init__.py` | 0 | package marker |
| `backend/chatbot/client.py` | ~70 | Token Factory client, `stream_completion()`, `is_configured()` |
| `backend/chatbot/context.py` | ~180 | system prompt constant, template-source reader (`lru_cache`), audit/log-tail readers, loss-snapshot formatter, `assemble_messages()` |
| `backend/api/chatbot.py` | ~100 | the two routes (§7) |
| `backend/db.py` (existing) | +35 | `chat_messages` table + `add_chat_message()` + `get_chat_messages(limit=None)` |
| `backend/logging_config.py` (existing) | +5 | `chatbot_log` logger |
| `config/settings.py` (existing) | +8 | 4 new fields (§8) |
| `frontend/src/types.ts` (existing) | +10 | `ChatMessage` interface |
| `frontend/src/hooks/useChatStream.ts` | ~70 | fetch + `ReadableStream` SSE parser |
| `frontend/src/components/ChatPanel.tsx` | ~110 | message list, input, calls the hook |
| `frontend/src/App.tsx` (existing) | +15 | mount `<ChatPanel>` |
| **Total** | **~603** | |

If `context.py` grows past ~220 lines during implementation, split
audit/log-tail parsing out into `context_log.py` rather than let one file
absorb unrelated growth.

## 12. Testing

Integration test hitting `POST /api/chatbot/{id}/message` with a mocked
Token Factory client (no real API calls in test runs) — asserts SSE stream
shape and that both the user and assistant messages land in
`chat_messages` with correct roles. Per project convention, test cleans up
any experiment/chat rows it creates.

## 13. Explicit non-scope for v1

- No code editing by the chatbot.
- No chatbot-generated configs or runs.
- No cross-session log search.
- No arbitrary-codebase tool-calling / RAG.
- No prompt-caching cost optimization (Nebius doesn't bill for it yet — see §3).
