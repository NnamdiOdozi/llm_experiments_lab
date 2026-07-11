import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import WorkerIdleBanner from "./WorkerIdleBanner";
import * as api from "../hooks/useApi";

describe("WorkerIdleBanner", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders nothing when there is no worker", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "none", seconds_idle: null, idle_timeout_seconds: null, warning_seconds: null,
    });

    const { container } = render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(api.getWorkerStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the worker is ready and not within the warning window", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 60, idle_timeout_seconds: 1800, warning_seconds: 600,
    });

    const { container } = render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(api.getWorkerStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("shows a countdown warning when within the warning window", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 1290, idle_timeout_seconds: 1800, warning_seconds: 600,
    });

    render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(screen.getByText(/8m 30s/)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /continue session/i })).toBeInTheDocument();
  });

  it("sends a heartbeat and re-polls when Continue session is clicked", async () => {
    const getStatus = vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 1290, idle_timeout_seconds: 1800, warning_seconds: 600,
    });
    const heartbeat = vi.spyOn(api, "sendWorkerHeartbeat").mockResolvedValue({ ok: true });

    render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(screen.getByRole("button", { name: /continue session/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /continue session/i }));

    await waitFor(() => expect(heartbeat).toHaveBeenCalledWith("cpu"));
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("shows a stopped notice that can be dismissed", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "stopped", seconds_idle: 1900, idle_timeout_seconds: 1800, warning_seconds: 600,
    });

    render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(screen.getByText(/stopped due to inactivity/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(screen.queryByText(/stopped due to inactivity/i)).not.toBeInTheDocument();
  });
});
