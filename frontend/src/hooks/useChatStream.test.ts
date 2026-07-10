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
      { id: 1, experiment_id: 5, role: "user" as const, content: "hi", prompt_tokens: null, completion_tokens: null, total_tokens: null, latency_ms: null, created_at: "2026-01-01" },
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
      vi.fn((_url: string, opts?: RequestInit) => {
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
      vi.fn((_url: string, opts?: RequestInit) => {
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
      vi.fn((_url: string, opts?: RequestInit) => {
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
