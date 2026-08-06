import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import PausePrompt from "./PausePrompt";
import * as api from "../hooks/useApi";
import type { DiagnosticSnapshot, DiagnosticSessionResponse } from "../types";

function snapshot(step: number, char: string): DiagnosticSnapshot {
  return {
    schema_version: 1,
    diagnostic_session_id: "diag-test",
    generation_step: step,
    input_tokens: [{ position: 0, id: 46, text: "h" }],
    generated_token: { position: step, id: 1, text: char },
    nodes: {},
    attention: { available: false, reason: "Not requested" },
    activation_summaries: { available: false, reason: "Not requested" },
    lm_head: { logits_shape: [1, 1, 65], selected_position: 0, top_k: [], top_k_by_position: [] },
    position_tokens: [],
    complete: true,
  };
}

const baseProps = {
  runId: 1,
  canPrompt: true,
  attentionBlock: null,
  attentionHead: null,
  showQKVDetail: false,
  attentionWindowOffset: 0,
  nodeWindowOffset: 0,
  temperature: 0.8,
  decodingMode: "sample",
};

describe("PausePrompt", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("does not deadlock >/>> after a mixed >-then->> session reaches max_new_tokens — a new prompt can still be run", async () => {
    // Real bug report, 2026-07-14: single-stepped once, then hit >> to
    // finish — after that, both > and >> stayed permanently disabled even
    // after typing a brand new prompt. Root cause: atCap (which disables
    // both buttons) was computed from diagnosticStep alone, and
    // diagnosticStep is deliberately left at its final value after a
    // session closes (so the reached count stays visible) — but the ONLY
    // code path that resets diagnosticStep back to 0 is ensureSession(),
    // called from inside the very button handlers atCap was disabling. A
    // genuine deadlock with no way out short of a page reload. Fixed by
    // requiring an open session for atCap to be true. See
    // docs/DESIGN_DECISIONS.md.
    vi.spyOn(api, "startDiagnostic")
      .mockResolvedValueOnce(
        { diagnostic_session_id: "diag-1", tokens: [{ position: 0, id: 46, text: "h" }] } as DiagnosticSessionResponse,
      )
      .mockResolvedValueOnce(
        { diagnostic_session_id: "diag-2", tokens: [{ position: 0, id: 46, text: "w" }] } as DiagnosticSessionResponse,
      );
    vi.spyOn(api, "stepDiagnostic")
      .mockResolvedValueOnce(snapshot(1, "e"))
      .mockResolvedValueOnce(snapshot(1, "o"));
    vi.spyOn(api, "generateDiagnosticStream").mockImplementation(async function* () {
      yield { position: 2, id: 2, text: "l", generation_step: 2 };
      yield { final_snapshot: snapshot(2, "l") };
    });

    const { container } = render(<PausePrompt {...baseProps} maxNewTokens={2} />);

    // Type a prompt, single-step once (1 of 2 max_new_tokens).
    fireEvent.change(screen.getByPlaceholderText("Enter a prompt..."), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: ">" }));
    await waitFor(() => expect(screen.getByText(/Step 1 of 2/)).toBeInTheDocument());
    expect(container.querySelector("pre")).toHaveStyle({ fontSize: "14px" });

    // Then >> to finish the remaining budget (1 more token -> reaches cap, session auto-closes).
    fireEvent.click(screen.getByRole("button", { name: ">>" }));
    await waitFor(() => expect(screen.getByText(/Step 2 of 2/)).toBeInTheDocument());

    // Prompt box must be editable again (session closed).
    const input = screen.getByPlaceholderText("Enter a prompt...") as HTMLInputElement;
    expect(input).not.toBeDisabled();

    // The actual bug: type a brand new prompt — > must not stay stuck disabled.
    fireEvent.change(input, { target: { value: "world" } });
    expect(screen.getByRole("button", { name: ">" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: ">>" })).not.toBeDisabled();

    // And it must actually still work — a fresh session starts and steps.
    fireEvent.click(screen.getByRole("button", { name: ">" }));
    await waitFor(() => expect(api.startDiagnostic).toHaveBeenCalledTimes(2));
  });
});
