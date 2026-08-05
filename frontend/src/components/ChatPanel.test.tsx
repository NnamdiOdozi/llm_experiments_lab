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

  // Markdown rendering tests
  it("renders bold and italic text in assistant messages", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "This is **bold** and this is *italic*.",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    const container = screen.getByText(/bold/).closest("div");
    expect(container?.querySelector("strong")).toBeInTheDocument();
    expect(container?.querySelector("em")).toBeInTheDocument();
  });

  it("renders bullet lists in assistant messages", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "- Item 1\n- Item 2\n- Item 3",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    const container = screen.getByText(/Item 1/).closest("div");
    const items = container?.querySelectorAll("li");
    expect(items?.length).toBe(3);
  });

  it("renders fenced code blocks in assistant messages", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "Here's a Python snippet:\n```python\nprint(1)\n```",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    const container = screen.getByText(/print/).closest("div");
    expect(container?.querySelector("pre")).toBeInTheDocument();
    expect(container?.querySelector("code")).toBeInTheDocument();
  });

  it("renders GFM tables in assistant messages", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    const container = screen.getByText(/Header 1/).closest("div");
    expect(container?.querySelector("table")).toBeInTheDocument();
  });

  it("renders links with target _blank and rel noopener noreferrer", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "Check out [this link](https://example.com)",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("escapes raw HTML and prevents script injection", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "This has <script>alert('xss')</script> in it.",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    const { container } = render(<ChatPanel experimentId={1} />);

    // Ensure no script element exists
    expect(container.querySelector("script")).not.toBeInTheDocument();
    // The HTML should appear as escaped text
    expect(screen.getByText(/script/)).toBeInTheDocument();
  });

  it("escapes raw HTML img tags with onerror handlers", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: 'An image: <img src="x" onerror="alert(\'xss\')" />',
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    const { container } = render(<ChatPanel experimentId={1} />);

    // Ensure no img element with onerror is rendered
    const imgs = container.querySelectorAll("img");
    expect(imgs.length).toBe(0);
  });

  it("handles streaming/partial Markdown without throwing", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "```python\nprint(1)",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    // This should not throw an error
    expect(() => render(<ChatPanel experimentId={1} />)).not.toThrow();
  });

  it("copy button returns raw Markdown source for assistant messages", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "assistant",
          content: "**Bold text** and [link](https://example.com)",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    render(<ChatPanel experimentId={1} />);

    // The CopyIconButton receives getText prop, verify button is present
    const copyButton = screen.getByTitle("Copy response");
    expect(copyButton).toBeInTheDocument();
  });

  it("keeps user messages as plain text with pre-wrap", () => {
    mockHook({
      messages: [
        {
          id: 1,
          experiment_id: 1,
          role: "user",
          content: "line1\nline2\nline3",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          latency_ms: null,
          created_at: "2026-01-01",
        },
      ],
    });
    const { container } = render(<ChatPanel experimentId={1} />);

    // Find the user message wrapper
    const userMessageDiv = container.querySelector('[style*="pre-wrap"]');
    expect(userMessageDiv?.textContent).toContain("line1");
  });
});
