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
    clearMessages: vi.fn(),
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

  it("shows no Clear chat button when there's no history yet", () => {
    mockHook({ messages: [] });
    render(<ChatPanel experimentId={1} />);
    expect(screen.queryByText(/clear chat/i)).not.toBeInTheDocument();
  });

  it("Clear chat requires a second click to confirm before actually clearing", () => {
    const clearMessages = vi.fn();
    mockHook({
      messages: [
        { id: 1, experiment_id: 1, role: "user", content: "hi", prompt_tokens: null, completion_tokens: null, total_tokens: null, latency_ms: null, created_at: "2026-01-01" },
      ],
      clearMessages,
    });
    render(<ChatPanel experimentId={1} />);

    const clearButton = screen.getByRole("button", { name: /clear chat/i });
    fireEvent.click(clearButton);
    expect(clearMessages).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /click to confirm/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /click to confirm/i }));
    expect(clearMessages).toHaveBeenCalledTimes(1);
  });

  it("shows the error message when present", () => {
    mockHook({ error: "Token Factory timed out" });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByText("Token Factory timed out")).toBeInTheDocument();
  });

  it("disables the send button while loading", () => {
    mockHook({ loading: true });
    render(<ChatPanel experimentId={1} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("submits on Enter but not on Shift+Enter", () => {
    const sendMessage = mockHook();
    render(<ChatPanel experimentId={1} />);

    const input = screen.getByPlaceholderText(/ask about this run/i);
    fireEvent.change(input, { target: { value: "hello" } });

    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(sendMessage).not.toHaveBeenCalled();

    fireEvent.keyDown(input, { key: "Enter" });
    expect(sendMessage).toHaveBeenCalledWith("hello");
  });
});
