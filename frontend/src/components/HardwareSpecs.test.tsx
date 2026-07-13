import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import HardwareSpecs from "./HardwareSpecs";
import * as api from "../hooks/useApi";

function status(overrides: Partial<import("../hooks/useApi").WorkerStatus> = {}) {
  return {
    worker_status: "none",
    seconds_idle: null,
    idle_timeout_seconds: null,
    warning_seconds: null,
    backend_mode: "local",
    preset: null,
    actual_platform: null,
    configured_platform: "cpu-d3",
    configured_preset: "16vcpu-64gb",
    ...overrides,
  };
}

describe("HardwareSpecs", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the configured spec before any endpoint has run", async () => {
    vi.spyOn(api, "getWorkerStatus").mockImplementation((device: string) =>
      Promise.resolve(
        device === "cuda"
          ? status({ configured_platform: "gpu-l40s-a", configured_preset: "1gpu-8vcpu-32gb" })
          : status(),
      ),
    );

    render(<HardwareSpecs />);

    await waitFor(() => expect(screen.getByText(/cpu-d3/)).toBeInTheDocument());
    expect(screen.getByText(/16vcpu-64gb/)).toBeInTheDocument();
    expect(screen.getAllByText(/configured/).length).toBe(2);
    expect(screen.getByText(/gpu-l40s-a/)).toBeInTheDocument();
  });

  it("prefers the actual live spec once an endpoint has run", async () => {
    vi.spyOn(api, "getWorkerStatus").mockImplementation((device: string) =>
      Promise.resolve(
        device === "cuda"
          ? status({
              configured_platform: "gpu-l40s-a",
              configured_preset: "1gpu-8vcpu-32gb",
              actual_platform: "gpu-h100-a",
              preset: "1gpu-16vcpu-64gb",
            })
          : status(),
      ),
    );

    render(<HardwareSpecs />);

    await waitFor(() => expect(screen.getByText(/gpu-h100-a/)).toBeInTheDocument());
    expect(screen.getByText(/1gpu-16vcpu-64gb/)).toBeInTheDocument();
    expect(screen.getAllByText(/live/).length).toBeGreaterThan(0);
  });
});
