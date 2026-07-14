import { describe, it, expect, vi, afterEach } from "vitest";
import { generateDiagnosticStream, peekDiagnostic, updateConfig, ApiError, fetchRunStatus } from "./useApi";
import { ExperimentConfig } from "../types";

function sseResponse(frames: string[]): Response {
  const body = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

describe("generateDiagnosticStream", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends attention_layer/attention_head/qkv_detail in the request body when given", async () => {
    // Regression test: >> previously only ever sent max_new_tokens, silently
    // ignoring whatever block/head/QKV-detail was selected in Inspector even
    // though the backend route (DiagnosticsGenerateRequest) always accepted
    // them. See docs/DESIGN_DECISIONS.md.
    const fetchMock = vi.fn((_url: string, _opts?: RequestInit) =>
      Promise.resolve(sseResponse([`event: done\ndata: {"final_snapshot": {}}\n\n`]))
    );
    vi.stubGlobal("fetch", fetchMock);

    const gen = generateDiagnosticStream(7, "diag-abc", 20, {
      attention_layer: 2,
      attention_head: 3,
      qkv_detail: true,
    });
    for await (const _ of gen) {
      // drain
    }

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/training/7/diagnostics/diag-abc/generate");
    const body = JSON.parse(opts!.body as string);
    expect(body).toEqual({
      max_new_tokens: 20,
      attention_layer: 2,
      attention_head: 3,
      qkv_detail: true,
    });
  });

  it("omits attention params when not given", async () => {
    const fetchMock = vi.fn((_url: string, _opts?: RequestInit) =>
      Promise.resolve(sseResponse([`event: done\ndata: {"final_snapshot": {}}\n\n`]))
    );
    vi.stubGlobal("fetch", fetchMock);

    const gen = generateDiagnosticStream(7, "diag-abc", 20);
    for await (const _ of gen) {
      // drain
    }

    const [, opts] = fetchMock.mock.calls[0];
    const body = JSON.parse(opts!.body as string);
    expect(body).toEqual({ max_new_tokens: 20 });
  });

  it("throws on an event: error frame instead of silently ending the stream", async () => {
    // Real bug, 2026-07-15: "event: error" frames were parsed but never
    // yielded — a real mid->> failure looked identical to success (the
    // loop just ended, nothing updated, no explanation surfaced to the
    // user). See docs/DESIGN_DECISIONS.md.
    const fetchMock = vi.fn((_url: string, _opts?: RequestInit) =>
      Promise.resolve(
        sseResponse([
          `event: token\ndata: {"position": 8, "id": 1, "text": "a", "generation_step": 1}\n\n`,
          `event: error\ndata: {"error": "Diagnostic session not found"}\n\n`,
        ])
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const gen = generateDiagnosticStream(7, "diag-abc", 20);
    async function drain() {
      for await (const _ of gen) {
        // drain
      }
    }

    await expect(drain()).rejects.toThrow("Diagnostic session not found");
  });
});

describe("api() error handling (via updateConfig)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const config: ExperimentConfig = {
    template: "transformer",
    model: { block_size: 128 },
    training: {},
    inference: { max_new_tokens: 150, temperature: 0.8, decoding_mode: "sample" },
  };

  it("surfaces the backend's HTTPException detail, not just the status line", async () => {
    // Real bug, 2026-07-15: api() threw only "{status} {statusText}" (e.g.
    // "400 Bad Request"), discarding FastAPI's actual detail message —
    // every rejected call in the app lost its specific reason. See
    // docs/DESIGN_DECISIONS.md.
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ detail: "max_new_tokens (150) cannot exceed block_size (128)" }), {
          status: 400,
          statusText: "Bad Request",
        })
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(updateConfig(1, config)).rejects.toThrow(
      "max_new_tokens (150) cannot exceed block_size (128)"
    );
  });

  it("falls back to the status line when the body isn't JSON", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(new Response("not json", { status: 500, statusText: "Internal Server Error" }))
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(updateConfig(1, config)).rejects.toThrow("500 Internal Server Error");
  });

  it("throws an ApiError carrying the HTTP status, even when the message is a detail string", async () => {
    // Real bug, 2026-07-14: App.tsx's disconnect heuristic classified
    // network-vs-HTTP by matching the MESSAGE against /^4\d\d/ — the
    // moment api() started throwing detail strings ("Run not found"),
    // every 4xx-with-detail was misread as a network failure and tripped
    // a false "Backend disconnected" banner. The status must survive as
    // a real field so callers never have to parse message text. See
    // docs/DESIGN_DECISIONS.md.
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ detail: "Run not found" }), { status: 404, statusText: "Not Found" })
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const err = await fetchRunStatus(999).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.message).toBe("Run not found");
  });
});

describe("peekDiagnostic", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts to the /peek route with the given attention params", async () => {
    // New route (2026-07-14) for auto-refreshing attention when Head/Block
    // changes in Inspector, without a full step click. See
    // docs/DESIGN_DECISIONS.md.
    const fetchMock = vi.fn((_url: string, _opts?: RequestInit) =>
      Promise.resolve(new Response(JSON.stringify({ generation_step: 1 }), { status: 200 }))
    );
    vi.stubGlobal("fetch", fetchMock);

    await peekDiagnostic(7, "diag-abc", { attention_layer: 2, attention_head: 3, qkv_detail: true });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/training/7/diagnostics/diag-abc/peek");
    expect(opts!.method).toBe("POST");
    const body = JSON.parse(opts!.body as string);
    expect(body).toEqual({ attention_layer: 2, attention_head: 3, qkv_detail: true });
  });
});
