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
    if (bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
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
    <div className="panel" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <h3>Lab Assistant</h3>
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
            {m.content}
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
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
