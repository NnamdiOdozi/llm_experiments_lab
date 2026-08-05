import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import WorkerIdleBanner from "./WorkerIdleBanner";
import * as api from "../hooks/useApi";

describe("WorkerIdleBanner", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders nothing when there is no worker", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "none", seconds_idle: null, idle_timeout_seconds: null, warning_seconds: null,
      backend_mode: "local", preset: null,
    });

    const { container } = render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(api.getWorkerStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the worker is ready and not within the warning window", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 60, idle_timeout_seconds: 1800, warning_seconds: 600,
      backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
    });

    const { container } = render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(api.getWorkerStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("shows a countdown warning when within the warning window", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 1290, idle_timeout_seconds: 1800, warning_seconds: 600,
      backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
    });

    render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(screen.getByText(/8m 30s/)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /continue session/i })).toBeInTheDocument();
  });

  it("sends a heartbeat and re-polls when Continue session is clicked", async () => {
    const getStatus = vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 1290, idle_timeout_seconds: 1800, warning_seconds: 600,
      backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
    });
    const heartbeat = vi.spyOn(api, "sendWorkerHeartbeat").mockResolvedValue({ ok: true });

    render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(screen.getByRole("button", { name: /continue session/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /continue session/i }));

    // sendWorkerHeartbeat now takes device and optional gpuFlavor parameter
    await waitFor(() => expect(heartbeat).toHaveBeenCalledWith("cpu", undefined));
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("shows a stopped notice (that can be dismissed) once it's actually seen the worker go ready then stop", async () => {
    // Real bug report, 2026-07-14: this banner appeared on a cold morning
    // load, before the worker had ever been polled as "ready" this
    // session — read as a stale alarm about someone else's inactivity
    // rather than a heads-up about something the user just watched
    // happen. The banner should only show once this component itself has
    // observed the ready -> stopped transition. See docs/DESIGN_DECISIONS.md.
    vi.useFakeTimers();
    try {
      vi.spyOn(api, "getWorkerStatus")
        .mockResolvedValueOnce({
          worker_status: "ready", seconds_idle: 60, idle_timeout_seconds: 1800, warning_seconds: 600,
          backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
        })
        .mockResolvedValue({
          worker_status: "stopped", seconds_idle: 1900, idle_timeout_seconds: 1800, warning_seconds: 600,
          backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
        });

      render(<WorkerIdleBanner device="cpu" />);

      // Flush the initial mount poll (resolves "ready") — nothing shown yet.
      await act(() => vi.advanceTimersByTimeAsync(0));
      expect(screen.queryByText(/stopped due to inactivity/i)).not.toBeInTheDocument();

      // Advance past one poll interval — next poll resolves "stopped".
      // POLL_INTERVAL_MS in the component is 15000; kept in sync here.
      await act(() => vi.advanceTimersByTimeAsync(15000));
      expect(screen.getByText(/stopped due to inactivity/i)).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
      expect(screen.queryByText(/stopped due to inactivity/i)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not show the stopped notice on a cold load that never saw the worker ready", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "stopped", seconds_idle: 1900, idle_timeout_seconds: 1800, warning_seconds: 600,
      backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
    });

    const { container } = render(<WorkerIdleBanner device="cpu" />);

    await waitFor(() => expect(api.getWorkerStatus).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText(/stopped due to inactivity/i)).not.toBeInTheDocument();
  });
});
