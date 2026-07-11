import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import WorkerPanel from "./WorkerPanel";
import * as api from "../hooks/useApi";

describe("WorkerPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a placeholder when no run has started", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "none", seconds_idle: null, idle_timeout_seconds: null,
      warning_seconds: null, backend_mode: "local", preset: null,
    });

    render(<WorkerPanel runId={null} device="cpu" />);

    await waitFor(() => expect(screen.getByText(/no events yet/i)).toBeInTheDocument());
  });

  it("shows step/loss event lines from metrics", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "none", seconds_idle: null, idle_timeout_seconds: null,
      warning_seconds: null, backend_mode: "local", preset: null,
    });
    vi.spyOn(api, "fetchRunStatus").mockResolvedValue({
      run_id: 5, status: "running", current_step: 20, total_steps: 100, metrics_count: 1, template: "transformer", elapsed_seconds: 10,
    });
    vi.spyOn(api, "fetchMetrics").mockResolvedValue([
      { step: 20, train_loss: 2.9688, val_loss: 3.0093 },
    ]);

    render(<WorkerPanel runId={5} device="cpu" />);

    await waitFor(() => expect(screen.getByText(/step=20/)).toBeInTheDocument());
    expect(screen.getByText(/loss=2.9688/)).toBeInTheDocument();
  });

  it("shows an N/A message on the Raw Logs tab in local mode", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "none", seconds_idle: null, idle_timeout_seconds: null,
      warning_seconds: null, backend_mode: "local", preset: null,
    });

    render(<WorkerPanel runId={null} device="cpu" />);
    fireEvent.click(screen.getByRole("button", { name: /raw logs/i }));

    await waitFor(() => expect(screen.getByText(/running locally/i)).toBeInTheDocument());
  });

  it("shows fetched log text on the Raw Logs tab in remote mode", async () => {
    vi.spyOn(api, "getWorkerStatus").mockResolvedValue({
      worker_status: "ready", seconds_idle: 5, idle_timeout_seconds: 1800,
      warning_seconds: 600, backend_mode: "nebius_endpoint", preset: "8vcpu-32gb",
    });
    vi.spyOn(api, "getWorkerLogs").mockResolvedValue({ logs: "INFO Uvicorn running\n" });

    render(<WorkerPanel runId={null} device="cpu" />);
    fireEvent.click(screen.getByRole("button", { name: /raw logs/i }));

    await waitFor(() => expect(screen.getByText(/Uvicorn running/)).toBeInTheDocument());
  });
});
