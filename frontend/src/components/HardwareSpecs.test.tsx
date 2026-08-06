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
    vi.spyOn(api, "getWorkerStatus").mockImplementation((device: string, _gpuFlavor?: string) =>
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
    // GPU preset should be decoded to human-readable format
    expect(screen.getByText(/48 GB VRAM/)).toBeInTheDocument();
  });

  it("prefers the actual live spec once an endpoint has run", async () => {
    vi.spyOn(api, "getWorkerStatus").mockImplementation((device: string, _gpuFlavor?: string) =>
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
    // Live actual preset should be decoded: "1 GPU · 16 vCPU · 64 GB RAM"
    expect(screen.getByText(/1 GPU.*16 vCPU.*64 GB RAM/)).toBeInTheDocument();
    // H100 has 80GB VRAM
    expect(screen.getByText(/80 GB VRAM/)).toBeInTheDocument();
    expect(screen.getAllByText(/live/).length).toBeGreaterThan(0);
  });

  it("passes gpuFlavor to getWorkerStatus when provided", async () => {
    const mockGetWorkerStatus = vi.spyOn(api, "getWorkerStatus").mockImplementation((device: string, gpuFlavor?: string) =>
      Promise.resolve(
        device === "cuda"
          ? status({
              configured_platform: gpuFlavor === "h100" ? "gpu-h100-sxm" : "gpu-l40s-a",
              configured_preset: gpuFlavor === "h100" ? "1gpu-16vcpu-200gb" : "1gpu-8vcpu-32gb",
            })
          : status(),
      ),
    );

    render(<HardwareSpecs gpuFlavor="h100" />);

    await waitFor(() => {
      expect(mockGetWorkerStatus).toHaveBeenCalledWith("cuda", "h100");
    });
    expect(screen.getByText(/gpu-h100-sxm/)).toBeInTheDocument();
  });
});
