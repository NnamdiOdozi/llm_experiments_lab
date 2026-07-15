import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import OpenRunsPage from "./OpenRunsPage";
import * as api from "../hooks/useApi";

describe("OpenRunsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a message when there are no open runs", async () => {
    vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([]);

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);

    await waitFor(() => expect(screen.getByText(/no active runs/i)).toBeInTheDocument());
  });

  it("lists open runs with experiment name, device, and status", async () => {
    vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([
      {
        id: 42, experiment_id: 7, experiment_name: "My Cool Experiment", status: "running",
        device: "cpu", execution_backend: "local", current_step: 10, total_steps: 100, started_at: "2026-07-11 22:00:00",
      },
    ]);

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);

    await waitFor(() => expect(screen.getByText("My Cool Experiment")).toBeInTheDocument());
    expect(screen.getByText(/run #42/i)).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("stops a run and removes it from the list on click", async () => {
    vi.spyOn(api, "fetchOpenRuns")
      .mockResolvedValueOnce([
        {
          id: 42, experiment_id: 7, experiment_name: "My Cool Experiment", status: "running",
          device: "cpu", execution_backend: "local", current_step: 10, total_steps: 100, started_at: "2026-07-11 22:00:00",
        },
      ])
      .mockResolvedValueOnce([]);
    const stopTraining = vi.spyOn(api, "stopTraining").mockResolvedValue({ run_id: 42 });

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);

    await waitFor(() => expect(screen.getByText("My Cool Experiment")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /stop/i }));

    await waitFor(() => expect(stopTraining).toHaveBeenCalledWith(42));
    await waitFor(() => expect(screen.getByText(/no active runs/i)).toBeInTheDocument());
  });

  it("calls onClose when the back button is clicked", async () => {
    vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([]);
    const onClose = vi.fn();

    render(<OpenRunsPage onClose={onClose} onReopen={() => {}} />);

    await waitFor(() => expect(screen.getByText(/no active runs/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /back/i }));

    expect(onClose).toHaveBeenCalled();
  });

  it("calls onReopen with the run when Open is clicked", async () => {
    const run = {
      id: 42, experiment_id: 7, experiment_name: "My Cool Experiment", status: "paused",
      device: "cuda", execution_backend: "nebius_endpoint", current_step: 10, total_steps: 100, started_at: "2026-07-11 22:00:00",
    };
    vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([run]);
    const onReopen = vi.fn();

    render(<OpenRunsPage onClose={() => {}} onReopen={onReopen} />);

    await waitFor(() => expect(screen.getByText("My Cool Experiment")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /^open$/i }));

    expect(onReopen).toHaveBeenCalledWith(run);
  });

  // Direct user request, 2026-07-15: no way back to a run once it finished/
  // failed/stopped. See docs/DESIGN_DECISIONS.md §79b.
  it("defaults to the Active filter without requesting terminal runs", async () => {
    const fetchOpenRuns = vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([]);

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);

    await waitFor(() => expect(fetchOpenRuns).toHaveBeenCalledWith(false));
  });

  it("requests terminal runs and shows a completed run when the Completed filter is clicked", async () => {
    const fetchOpenRuns = vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([
      {
        id: 42, experiment_id: 7, experiment_name: "Finished Run", status: "completed",
        device: "cpu", execution_backend: "local", current_step: 100, total_steps: 100, started_at: "2026-07-11 22:00:00",
      },
    ]);

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Completed" }));

    await waitFor(() => expect(fetchOpenRuns).toHaveBeenCalledWith(true));
    await waitFor(() => expect(screen.getByText("Finished Run")).toBeInTheDocument());
  });

  it("hides the Stop button for a terminal run but keeps Open", async () => {
    vi.spyOn(api, "fetchOpenRuns").mockResolvedValue([
      {
        id: 42, experiment_id: 7, experiment_name: "Finished Run", status: "completed",
        device: "cpu", execution_backend: "local", current_step: 100, total_steps: 100, started_at: "2026-07-11 22:00:00",
      },
    ]);

    render(<OpenRunsPage onClose={() => {}} onReopen={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Completed" }));

    await waitFor(() => expect(screen.getByText("Finished Run")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /^stop$/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^open$/i })).toBeInTheDocument();
  });
});
