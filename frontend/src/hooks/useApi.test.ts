import { describe, it, expect, vi, afterEach } from "vitest";
import { generateDiagnosticStream, peekDiagnostic } from "./useApi";

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
