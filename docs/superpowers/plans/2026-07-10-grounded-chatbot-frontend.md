# Grounded Chatbot — Frontend Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frontend chat panel that calls the already-merged backend (`GET/POST /api/chatbot/{id}/...`, SSE streaming) — a message list, an input box, and the SSE-consuming hook, mounted into the existing app layout.

**Architecture:** A `useChatStream` hook owns SSE parsing (fetch + `ReadableStream`, since browsers don't support `EventSource` with a POST body) and message state; `ChatPanel` is a thin render layer over it, following the existing `ExperimentNotes`/`PausePrompt` component pattern (inline styles, `.panel`/`.btn-primary` CSS classes, no external UI library).

**Tech Stack:** React 18 + TypeScript (existing), Vitest + `@testing-library/react` + jsdom (new — no frontend test framework currently exists in this repo, mirroring the backend's Plan A precedent).

**Working directory for all steps:** `.worktrees/grounded-chatbot-frontend/frontend/` (branch `feature/grounded-chatbot-frontend`).

---

### Task 1: Test harness setup

No frontend test framework exists yet. This adds Vitest + Testing Library and proves it works before anything else is built on top of it.

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/test/smoke.test.tsx`

- [ ] **Step 1: Add test dependencies**

Run from `frontend/`:
```bash
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 2: Add the test script**

In `frontend/package.json`, add to `"scripts"`:
```json
"test": "vitest run"
```

- [ ] **Step 3: Create the Vitest config**

Create `frontend/vitest.config.ts`:
```typescript
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
```

- [ ] **Step 4: Create the setup file**

Create `frontend/src/test/setup.ts`:
```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 5: Create a smoke test**

Create `frontend/src/test/smoke.test.tsx`:
```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

describe("smoke", () => {
  it("renders a div", () => {
    render(<div>hello</div>);
    expect(screen.getByText("hello")).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run it**

Run: `npm test`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add package.json package-lock.json vitest.config.ts src/test/setup.ts src/test/smoke.test.tsx
git commit -m "test: add vitest + testing-library harness"
```

---

### Task 2: `ChatMessage` type

**Files:**
- Modify: `frontend/src/types.ts`

- [ ] **Step 1: Add the type**

Append to `frontend/src/types.ts` (matches the backend's `chat_messages` table columns exactly — see `backend/db.py` and `docs/superpowers/specs/2026-07-10-grounded-chatbot-design.md` §6):

```typescript
export interface ChatMessage {
  id: number;
  experiment_id: number;
  role: "user" | "assistant";
  content: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  latency_ms: number | null;
  created_at: string;
}
```

- [ ] **Step 2: Verify it type-checks**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/types.ts
git commit -m "feat: add ChatMessage type"
```

---

### Task 3: `useChatStream` hook

**Files:**
- Create: `frontend/src/hooks/useChatStream.ts`
- Test: `frontend/src/hooks/useChatStream.test.ts`

This hook owns all SSE parsing logic. It's the highest-risk piece of Plan B (hand-rolled stream parsing, no library), so it gets the most test coverage.

**API contract this hook talks to** (already live on `main`, from Plan A — do not modify the backend):
- `GET /api/chatbot/{id}/messages` → `ChatMessage[]`
- `POST /api/chatbot/{id}/message` with `{message: string}` body → `text/event-stream` response. Frames: `data: {"delta": "..."}\n\n` (repeated), then either `event: done\ndata: {"usage": {...}}\n\n` or `event: error\ndata: {"error": "..."}\n\n`. Returns `503` with a JSON error body if the chatbot isn't configured (no `NEBIUS_KEY` on the backend).

- [ ] **Step 1: Write failing tests**

Create `frontend/src/hooks/useChatStream.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { useChatStream } from "./useChatStream";

function sseResponse(frames: string[], status = 200): Response {
  const body = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame));
      }
      controller.close();
    },
  });
  return new Response(body, { status });
}

describe("useChatStream", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.endsWith("/messages")) {
          return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
        }
        throw new Error(`unexpected fetch in beforeEach stub: ${url}`);
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads existing history on mount", async () => {
    const history = [
      { id: 1, experiment_id: 5, role: "user", content: "hi", prompt_tokens: null, completion_tokens: null, total_tokens: null, latency_ms: null, created_at: "2026-01-01" },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(JSON.stringify(history), { status: 200 })))
    );

    const { result } = renderHook(() => useChatStream(5));

    await waitFor(() => expect(result.current.messages).toHaveLength(1));
    expect(result.current.messages[0].content).toBe("hi");
  });

  it("streams assistant deltas into a growing message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, opts?: RequestInit) => {
        if (!opts) return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
        return Promise.resolve(
          sseResponse([
            `data: ${JSON.stringify({ delta: "Hello" })}\n\n`,
            `data: ${JSON.stringify({ delta: " world" })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ usage: { total_tokens: 10 } })}\n\n`,
          ])
        );
      })
    );

    const { result } = renderHook(() => useChatStream(5));
    await waitFor(() => expect(result.current.messages).toHaveLength(0));

    await act(async () => {
      await result.current.sendMessage("hi there");
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0]).toMatchObject({ role: "user", content: "hi there" });
    expect(result.current.messages[1]).toMatchObject({ role: "assistant", content: "Hello world" });
    expect(result.current.loading).toBe(false);
  });

  it("sets unavailable on a 503 response and does not add a phantom assistant message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, opts?: RequestInit) => {
        if (!opts) return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
        return Promise.resolve(new Response(JSON.stringify({ detail: "unavailable" }), { status: 503 }));
      })
    );

    const { result } = renderHook(() => useChatStream(5));
    await waitFor(() => expect(result.current.messages).toHaveLength(0));

    await act(async () => {
      await result.current.sendMessage("hi");
    });

    expect(result.current.unavailable).toBe(true);
    expect(result.current.messages.filter((m) => m.role === "assistant")).toHaveLength(0);
  });

  it("surfaces an error event from the stream", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, opts?: RequestInit) => {
        if (!opts) return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
        return Promise.resolve(
          sseResponse([`event: error\ndata: ${JSON.stringify({ error: "Token Factory timed out" })}\n\n`])
        );
      })
    );

    const { result } = renderHook(() => useChatStream(5));
    await waitFor(() => expect(result.current.messages).toHaveLength(0));

    await act(async () => {
      await result.current.sendMessage("hi");
    });

    expect(result.current.error).toBe("Token Factory timed out");
  });

  it("does not send an empty or whitespace-only message", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify([]), { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useChatStream(5));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1)); // just the initial history GET

    await act(async () => {
      await result.current.sendMessage("   ");
    });

    expect(fetchMock).toHaveBeenCalledTimes(1); // no additional POST
  });
});
```

- [ ] **Step 2: Run it, verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './useChatStream'`

- [ ] **Step 3: Implement the hook**

Create `frontend/src/hooks/useChatStream.ts`:

```typescript
import { useState, useEffect, useCallback } from "react";
import { ChatMessage } from "../types";

const BASE = "/api/chatbot";

let nextLocalId = -1;

function localMessage(experimentId: number, role: "user" | "assistant", content: string): ChatMessage {
  return {
    id: nextLocalId--,
    experiment_id: experimentId,
    role,
    content,
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    latency_ms: null,
    created_at: new Date().toISOString(),
  };
}

export function useChatStream(experimentId: number) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    fetch(`${BASE}/${experimentId}/messages`)
      .then((res) => res.json())
      .then(setMessages)
      .catch(() => {
        // History load failure isn't fatal — the panel still works for new messages
      });
  }, [experimentId]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setLoading(true);
      setError(null);
      setUnavailable(false);

      const userMsg = localMessage(experimentId, "user", text);
      setMessages((prev) => [...prev, userMsg]);

      try {
        const res = await fetch(`${BASE}/${experimentId}/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });

        if (res.status === 503) {
          setUnavailable(true);
          setLoading(false);
          return;
        }
        if (!res.ok || !res.body) {
          throw new Error(`${res.status} ${res.statusText}`);
        }

        const assistantMsg = localMessage(experimentId, "assistant", "");
        setMessages((prev) => [...prev, assistantMsg]);
        let assistantText = "";

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            if (!frame.trim()) continue;
            let event = "message";
            let data = "";
            for (const line of frame.split("\n")) {
              if (line.startsWith("event: ")) event = line.slice(7);
              if (line.startsWith("data: ")) data = line.slice(6);
            }
            if (!data) continue;
            const parsed = JSON.parse(data);

            if (event === "error") {
              setError(parsed.error ?? "Chat stream failed");
            } else if (event === "done") {
              // usage stats (parsed.usage) available here if ever surfaced in the UI
            } else {
              assistantText += parsed.delta ?? "";
              const finalText = assistantText;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsg.id ? { ...m, content: finalText } : m))
              );
            }
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Chat request failed");
      }
      setLoading(false);
    },
    [experimentId]
  );

  return { messages, sendMessage, loading, error, unavailable };
}
```

- [ ] **Step 4: Run it, verify it passes**

Run: `npm test`
Expected: `6 passed` (1 smoke + 5 in this file)

- [ ] **Step 5: Commit**

```bash
git add src/hooks/useChatStream.ts src/hooks/useChatStream.test.ts
git commit -m "feat: add useChatStream SSE hook"
```

---

### Task 4: `ChatPanel` component

**Files:**
- Create: `frontend/src/components/ChatPanel.tsx`
- Test: `frontend/src/components/ChatPanel.test.tsx`

Follows the same structure as `ExperimentNotes.tsx`/`PausePrompt.tsx`: a `.panel` div, inline styles, no external UI library. This task mocks `useChatStream` itself (already tested in Task 3) rather than re-testing SSE parsing.

- [ ] **Step 1: Write failing tests**

Create `frontend/src/components/ChatPanel.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatPanel from "./ChatPanel";
import * as useChatStreamModule from "../hooks/useChatStream";

function mockHook(overrides: Partial<ReturnType<typeof useChatStreamModule.useChatStream>> = {}) {
  const sendMessage = vi.fn();
  vi.spyOn(useChatStreamModule, "useChatStream").mockReturnValue({
    messages: [],
    sendMessage,
    loading: false,
    error: null,
    unavailable: false,
    ...overrides,
  });
  return sendMessage;
}

describe("ChatPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an unavailable message when the chatbot isn't configured", () => {
    mockHook({ unavailable: true });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });

  it("renders messages with role-based content", () => {
    mockHook({
      messages: [
        { id: 1, experiment_id: 1, role: "user", content: "what does this mean?", prompt_tokens: null, completion_tokens: null, total_tokens: null, latency_ms: null, created_at: "2026-01-01" },
        { id: 2, experiment_id: 1, role: "assistant", content: "It means loss is decreasing.", prompt_tokens: null, completion_tokens: null, total_tokens: null, latency_ms: null, created_at: "2026-01-01" },
      ],
    });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByText("what does this mean?")).toBeInTheDocument();
    expect(screen.getByText("It means loss is decreasing.")).toBeInTheDocument();
  });

  it("calls sendMessage with input text and clears the input on submit", () => {
    const sendMessage = mockHook();
    render(<ChatPanel experimentId={1} />);

    const input = screen.getByPlaceholderText(/ask about this run/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "why did loss spike?" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(sendMessage).toHaveBeenCalledWith("why did loss spike?");
    expect(input.value).toBe("");
  });

  it("does not call sendMessage for an empty input", () => {
    const sendMessage = mockHook();
    render(<ChatPanel experimentId={1} />);

    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("shows the error message when present", () => {
    mockHook({ error: "Token Factory timed out" });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByText("Token Factory timed out")).toBeInTheDocument();
  });

  it("disables the send button while loading", () => {
    mockHook({ loading: true });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByRole("button", { name: /\.\.\./ })).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run it, verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './ChatPanel'`

- [ ] **Step 3: Implement the component**

Create `frontend/src/components/ChatPanel.tsx`:

```tsx
import { useState, useRef, useEffect } from "react";
import { useChatStream } from "../hooks/useChatStream";

interface Props {
  experimentId: number;
}

export default function ChatPanel({ experimentId }: Props) {
  const { messages, sendMessage, loading, error, unavailable } = useChatStream(experimentId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit() {
    if (!input.trim() || loading) return;
    sendMessage(input);
    setInput("");
  }

  if (unavailable) {
    return (
      <div className="panel" style={{ opacity: 0.5 }}>
        <h3>Lab Assistant</h3>
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Chatbot unavailable — no Token Factory API key configured.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Lab Assistant</h3>
      <div
        style={{
          maxHeight: 300,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 8,
          marginBottom: 8,
        }}
      >
        {messages.map((m) => (
          <div
            key={m.id}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              background: m.role === "user" ? "var(--accent-dim)" : "var(--bg)",
              color: m.role === "user" ? "#fff" : "var(--text)",
              border: m.role === "assistant" ? "1px solid var(--border)" : "none",
              borderRadius: 8,
              padding: "6px 10px",
              maxWidth: "85%",
              fontSize: 13,
              whiteSpace: "pre-wrap",
            }}
          >
            {m.content}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      {error && <p style={{ fontSize: 12, color: "var(--red)" }}>{error}</p>}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this run..."
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          style={{ flex: 1 }}
        />
        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run it, verify it passes**

Run: `npm test`
Expected: `12 passed` (1 smoke + 5 hook + 6 component)

- [ ] **Step 5: Commit**

```bash
git add src/components/ChatPanel.tsx src/components/ChatPanel.test.tsx
git commit -m "feat: add ChatPanel component"
```

---

### Task 5: Mount `ChatPanel` in `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`

Mirrors the doc's own wireframe pairing ("Loss Curve + Chat" in the same panel group, §7 of the project discussion doc) — mounted in the main area, next to the loss chart.

- [ ] **Step 1: Add the import**

In `frontend/src/App.tsx`, add alongside the other component imports (after `ExperimentNotes`):

```typescript
import ChatPanel from "./components/ChatPanel";
```

- [ ] **Step 2: Mount it in the main area**

In the "Main area" `<div>` (the second column of the two-column grid), add `<ChatPanel experimentId={experimentId} />` right after the loss/drop-rate chart row and before `<ArchSchematic config={config} />`:

```tsx
        {/* Main area */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Loss + MoE drop rate charts side-by-side when MoE data present */}
          <div style={{ display: "flex", gap: 16 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <LossChart metrics={metrics} />
            </div>
            {metrics.some((m) => m.train_drop_rate != null) && (
              <div style={{ flex: 1, minWidth: 0 }}>
                <DropRateChart metrics={metrics} />
              </div>
            )}
          </div>
          <ChatPanel experimentId={experimentId} />
          <ArchSchematic config={config} />
```

(Everything below `<ArchSchematic ... />` in the existing file stays unchanged — this only inserts one new line.)

- [ ] **Step 3: Verify it builds and type-checks**

Run: `npm run build`
Expected: builds successfully, no TypeScript errors.

- [ ] **Step 4: Run the full test suite**

Run: `npm test`
Expected: `12 passed` (App.tsx has no new tests of its own — it's a one-line wiring change into an already-tested component; the existing manual smoke check in Task 6 covers it visually)

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx
git commit -m "feat: mount ChatPanel in the main app layout"
```

---

### Task 6: Manual smoke check against the real backend

Not a new automated test — confirms the panel actually renders and talks to the real (already-merged) backend from Plan A, including the "unavailable" state when no API key is configured.

- [ ] **Step 1: Start the backend** (from the main repo, not this worktree — Plan A is already merged to `main`)

```bash
cd /home/nodozi/projects/NEBIUS_MAR_2026/Nebius_serverless/llm_experiments_lab
uv run uvicorn backend.main:app --port 8000
```

- [ ] **Step 2: Start the frontend dev server** (from this worktree)

```bash
cd .worktrees/grounded-chatbot-frontend/frontend
npm run dev
```

- [ ] **Step 3: Verify in browser or via curl against the Vite dev server**

Open `http://localhost:5173`, pick a preset, confirm the "Lab Assistant" panel renders in the main area. Since `NEBIUS_KEY` isn't loaded in this shell (no `.env` in a fresh worktree checkout — same as Plan A's Task 8), expect the panel to show "Chatbot unavailable — no Token Factory API key configured." That confirms the full request path (browser → Vite proxy → FastAPI → 503) works end to end.

- [ ] **Step 4: Stop both servers**

`Ctrl-C` in both terminals.

No commit for this task — verification only.

---

## Definition of done for Plan B

- [ ] All tasks 1–5 committed on `feature/grounded-chatbot-frontend`
- [ ] `npm test` passes with 0 failures
- [ ] `npm run build` succeeds (type-checks + bundles)
- [ ] Task 6 manual check performed at least once
- [ ] Grounded chatbot feature is now end-to-end complete: backend (Plan A, merged) + frontend (this plan)
