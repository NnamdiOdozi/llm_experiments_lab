# PROJECT STATE — Session 2026-07-10/11 (Nebius CLI + grounded chatbot complete)

**Handoff written:** 2026-07-11 16:56 BST

## 1. Objective
LLM Experiments Lab: browser-based training tool (FastAPI + React/TS), submission for the
Nebius Serverless AI Builders Challenge. This session: (a) installed/configured the `nebius`
CLI for compute access, (b) designed and shipped a full-stack "grounded chatbot" feature — a
lab-assistant chatbot (Nebius Token Factory) grounded in the user's live experiment state
(config, loss trend, last change, architecture source, log tail).

## 2. Constraints
- POC scope by explicit instruction: minimal error handling, no auth, SQLite (not Postgres).
- No hardcoding — config lives in `config/settings.py` / `.env`. One deliberate exception:
  the chatbot's system-prompt text is a Python string constant, not externalized (it's prose
  defining behavior, not a deployment-dependent value — documented in the design spec §8).
- Never delete files without explicit confirmation; extra caution on `.db` files.
- This environment runs **Node 18** — forces older frontend test-tooling versions
  (vitest 0.34.6, jsdom 22.1.0; latest require Node 20+).
- Auto-mode classifier blocks destructive git ops (discarding local diffs, resolving merge
  conflicts by picking one side) unless the user explicitly names the target — expect to pause
  and ask, even when the resolution seems obviously safe.

## 3. File map
**Chatbot backend** (merged to `main`):
- `backend/chatbot/client.py` — Token Factory streaming client (`AsyncOpenAI` wrapper)
- `backend/chatbot/context.py` — grounding context assembly (static system prompt + template
  source + volatile loss/audit/log snapshot, static-first/volatile-last message ordering)
- `backend/api/chatbot.py` — `GET/POST /api/chatbot/{id}/...`, SSE streaming
- `backend/db.py` — `chat_messages` table + `add_chat_message()`/`get_chat_messages()`
- `config/settings.py:46-51` — `nebius_key`, `token_factory_base_url`, `token_factory_model`,
  `chatbot_log_tail_lines`, `chatbot_history_window_turns`

**Chatbot frontend** (merged to `main`):
- `frontend/src/hooks/useChatStream.ts` — hand-rolled SSE parser (fetch + ReadableStream)
- `frontend/src/components/ChatPanel.tsx` — chat UI, mounted in `App.tsx`'s main area
- `frontend/src/types.ts` — `ChatMessage` interface

**Docs:**
- `docs/superpowers/specs/2026-07-10-grounded-chatbot-design.md` — full design rationale
- `docs/superpowers/plans/2026-07-10-grounded-chatbot-{backend,frontend}.md` — TDD plans
- `docs/DESIGN_DECISIONS.md` §7 — audit-log substring-matching, documented as fragile-by-design

**Nebius CLI:**
- `scripts/install_nebius_cli.sh` — installer, non-interactive service-account auth
- Active profile: `mlflow-sa`, default project: `project-<redacted>`
  (default-project-eu-north1). `llm_experiments_lab` project also exists:
  `project-<redacted>` (not switched to — user chose to keep default).
- `~/.bashrc` has `export PATH="$HOME/.nebius/bin:$PATH"` (line ~240)

## 4. Commands
```bash
# Backend
cd /home/nodozi/projects/NEBIUS_MAR_2026/Nebius_serverless/llm_experiments_lab
uv run pytest -q                                    # 23 tests
uv run uvicorn backend.main:app --port 8000

# Frontend
cd frontend
npm test                                             # 12 tests
npm run build                                        # tsc + vite — run this, not just npm test
npm run dev                                          # Vite on 5173, proxies /api -> :8000

# Nebius CLI
nebius profile list / nebius iam whoami

# List live Token Factory models (don't trust blog/cookbook model names — verify live)
uv run python -c "
import httpx
from config.settings import settings
r = httpx.get(f'{settings.token_factory_base_url}models', headers={'Authorization': f'Bearer {settings.nebius_key}'})
for m in sorted(r.json()['data'], key=lambda x: x['id']): print(m['id'])
"
```

## 5. Open questions
- **Model choice**: currently `Qwen/Qwen3-Next-80B-A3B-Thinking` (3.9B active params). Compared
  against `moonshotai/Kimi-K2.7-Code` (32B active, coding-agent-specialized) — recommended
  keeping the general-reasoning model since the chatbot's job is theory-explanation with
  occasional code *reading*, not code *generation* (Moonshot's own docs say K2.7-Code trades
  general-purpose quality for coding-agent strength). Not finalized — user was deciding.
- **IP-protection framing** (raised 2026-07-11): current code-access scoping (template files
  only, not the full codebase) was designed for technical/security reasons — NOT IP protection.
  If the actual goal is preventing users from extracting source via chat, that's unbuilt
  (would need response-level filtering, not just input scoping).
- Phase 2 backlog item: agentic tool-calling for arbitrary codebase Q&A (e.g. "how does
  `runner.py`'s pause/resume work") — explicitly deferred, chatbot currently says "I don't have
  visibility into that" for anything outside config/loss/template-source/logs.
- Frontend `useChatStream.ts`'s `nextLocalId` module-level counter — fine for single-panel POC
  (documented in a code comment), would need per-instance state if a multi-panel UI is ever built.

## 6. What NOT to do / lessons learned
- **Write plan docs inside the worktree, not the main tree.** Happened twice this session
  (backend AND frontend plan docs both silently landed in the main repo's working directory
  instead of the worktree, went uncommitted, had to be recovered before merging). Confirm `pwd`
  before writing planning docs when a worktree is active.
- `.gitignore`'s old `tests/` blanket-ignore (forced every test-file commit to need `git add -f`)
  is now fixed (commit `9d5d45b`) — this was flagged as a known issue back in the 2026-06-28
  session too but never fixed until now.
- Something (cause not identified — a system note once described a `.gitignore` change as
  "intentional, don't mention it") commits directly to `main` mid-session, independent of normal
  worktree merges. Caused one real merge conflict (resolved safely — verified byte-identical
  content before taking one side). If it recurs, investigate the source before assuming safety.
- **Don't trust model names from blog posts/cookbooks — query the live `/v1/models` endpoint.**
  The original `token_factory_model` default (`Qwen3-235B-A22B-Thinking-2507`) came from a
  GitHub cookbook reference and doesn't actually exist on Token Factory. Only caught via a live
  end-to-end smoke test — the earlier isolated backend worktree had no `.env`, so it never
  actually reached the real API and this went unnoticed until the frontend smoke check.
- **`npm test` alone isn't enough for frontend TS changes — also run `npm run build`.** Vitest's
  transform doesn't enforce `tsconfig.json`'s `noUnusedParameters`; `tsc` (via `npm run build`)
  does. A latent violation from Task 3 wasn't caught until Task 5 ran the first full build.
- Test experiment rows (ids ~117-119) exist in the real `lab.db` from manual smoke tests —
  harmless, left alone (didn't touch the `.db` file directly, per the DB-caution rule).
- Don't background dev servers with plain `&` — use the `run_in_background` Bash param instead;
  and after killing one, verify with `ps aux` / `lsof -i :<port>`, not just `pgrep` (which can
  return stale PIDs for already-exited processes).
