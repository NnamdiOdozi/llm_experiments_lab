import { useState, useRef, useEffect, CSSProperties } from "react";
import { useChatStream } from "../hooks/useChatStream";
import { setChatMessageFeedback } from "../hooks/useApi";
import { ChatMessage } from "../types";

interface Props {
  experimentId: number;
}

const iconButtonStyle: CSSProperties = {
  background: "none",
  border: "none",
  color: "var(--text-dim)",
  cursor: "pointer",
  padding: 0,
  lineHeight: 0,
};

export default function ChatPanel({ experimentId }: Props) {
  const { messages, sendMessage, loading, error, unavailable, clearMessages } = useChatStream(experimentId);
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [feedbackOverrides, setFeedbackOverrides] = useState<Record<number, "up" | "down" | null>>({});
  const [confirmingClear, setConfirmingClear] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Two-click confirm (not a native confirm() dialog, to match the rest of
  // this app's UI) — resets a stuck/confused conversation without
  // touching the experiment or any of its runs. Direct user request,
  // 2026-07-14. See docs/DESIGN_DECISIONS.md.
  async function handleClearChat() {
    if (!confirmingClear) {
      setConfirmingClear(true);
      return;
    }
    setConfirmingClear(false);
    await clearMessages();
  }

  useEffect(() => {
    if (bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  function handleSubmit() {
    if (!input.trim() || loading) return;
    sendMessage(input);
    setInput("");
  }

  async function handleCopy(id: number, content: string) {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId((current) => (current === id ? null : current)), 1500);
  }

  function currentFeedback(m: ChatMessage): "up" | "down" | null {
    return feedbackOverrides[m.id] !== undefined ? feedbackOverrides[m.id] : m.feedback ?? null;
  }

  async function handleFeedback(m: ChatMessage, value: "up" | "down") {
    const next = currentFeedback(m) === value ? null : value;
    setFeedbackOverrides((prev) => ({ ...prev, [m.id]: next }));
    try {
      await setChatMessageFeedback(m.id, next);
    } catch {
      // Best-effort — a failed feedback PATCH isn't worth surfacing an error for
    }
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
    <div className="panel" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Lab Assistant</h3>
        {messages.length > 0 && (
          <button
            onClick={handleClearChat}
            onBlur={() => setConfirmingClear(false)}
            title="Delete this conversation's history — the experiment and its runs are untouched"
            style={{
              fontSize: 11,
              padding: "4px 8px",
              background: confirmingClear ? "var(--red)" : "none",
              color: confirmingClear ? "#fff" : "var(--text-dim)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            {confirmingClear ? "Click to confirm" : "Clear chat"}
          </button>
        )}
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
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
            {m.role === "assistant" && m.content === "" ? (
              <div className="typing-dots" style={{ display: "flex", gap: 3, padding: "4px 2px" }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--text-dim)" }} />
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--text-dim)" }} />
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--text-dim)" }} />
              </div>
            ) : (
              <div>{m.content}</div>
            )}
            {m.role === "assistant" && m.content !== "" && (
              <div style={{ display: "flex", gap: 10, paddingTop: 4 }}>
                <button
                  onClick={() => handleCopy(m.id, m.content)}
                  title="Copy response"
                  style={iconButtonStyle}
                >
                  {copiedId === m.id ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="9" y="9" width="11" height="11" rx="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                  )}
                </button>
                <button
                  onClick={() => handleFeedback(m, "up")}
                  title="Good response"
                  style={{
                    ...iconButtonStyle,
                    color: currentFeedback(m) === "up" ? "var(--green)" : "var(--text-dim)",
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
                    <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                  </svg>
                </button>
                <button
                  onClick={() => handleFeedback(m, "down")}
                  title="Bad response"
                  style={{
                    ...iconButtonStyle,
                    color: currentFeedback(m) === "down" ? "var(--red)" : "var(--text-dim)",
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
                    <path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
                  </svg>
                </button>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      {error && <p style={{ fontSize: 12, color: "var(--red)" }}>{error}</p>}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this run..."
          rows={4}
          onKeyDown={(e) => {
            // Enter sends; Shift+Enter inserts a newline (standard chat convention)
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          style={{
            width: "100%",
            resize: "vertical",
            fontSize: 13,
            fontFamily: "inherit",
            background: "var(--bg)",
            color: "var(--text)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: 8,
          }}
        />
        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
