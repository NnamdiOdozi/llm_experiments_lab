import { useState, useEffect, useCallback } from "react";
import { ChatMessage } from "../types";

const BASE = "/api/chatbot";

// Module-level counter for client-generated message IDs (negative, so they
// never collide with server-assigned positive IDs once persisted rows come
// back from GET .../messages). Shared across all useChatStream() instances —
// fine because only one ChatPanel is ever mounted at a time (see App.tsx).
// If a multi-panel UI is ever built, switch this to per-instance state.
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
