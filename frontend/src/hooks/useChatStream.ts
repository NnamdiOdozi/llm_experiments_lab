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
      .then((res) => {
        if (res.status === 404) {
          setUnavailable(true);
          return [];
        }
        if (!res.ok) return [];
        return res.json();
      })
      .then((data) => {
        setMessages(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        // History load failure isn't fatal — the panel still works for new messages
        setMessages([]);
      });
  }, [experimentId]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setLoading(true);
      setError(null);
      setUnavailable(false);

      const userMsg = localMessage(experimentId, "user", text);
      // Added immediately, not after the response arrives — this empty
      // placeholder is what ChatPanel renders the typing indicator for, so
      // it needs to exist for the whole request, not just once streaming
      // starts (that gap can itself be a second or more).
      const assistantMsg = localMessage(experimentId, "assistant", "");
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      function removePlaceholder() {
        setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id));
      }

      try {
        const res = await fetch(`${BASE}/${experimentId}/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });

        if (res.status === 503) {
          setUnavailable(true);
          removePlaceholder();
          setLoading(false);
          return;
        }
        if (!res.ok || !res.body) {
          removePlaceholder();
          throw new Error(`${res.status} ${res.statusText}`);
        }

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
              // Swap the local placeholder id for the real DB row id now
              // that the message is actually persisted — needed so
              // feedback (thumbs up/down) PATCHes a row that exists
              // instead of 404ing against an id the server never assigned.
              if (parsed.message_id != null) {
                const realId = parsed.message_id;
                setMessages((prev) =>
                  prev.map((m) => (m.id === assistantMsg.id ? { ...m, id: realId } : m))
                );
              }
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

  // Resets a stuck/confused conversation without touching the experiment
  // or any of its runs — direct user request, 2026-07-14. See
  // docs/DESIGN_DECISIONS.md.
  const clearMessages = useCallback(async () => {
    setError(null);
    try {
      await fetch(`${BASE}/${experimentId}/messages`, { method: "DELETE" });
      setMessages([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear chat");
    }
  }, [experimentId]);

  return { messages, sendMessage, loading, error, unavailable, clearMessages };
}
