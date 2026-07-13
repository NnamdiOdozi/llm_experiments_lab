import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import ExperimentBrowser from "./ExperimentBrowser";
import * as api from "../hooks/useApi";

function makeExperiment(overrides: Partial<import("../types").Experiment> = {}) {
  return {
    id: 3,
    name: "Tiny Shakespeare Run",
    config: { template: "transformer", model: {}, training: {} },
    notes_md: "",
    preset_key: "tiny-shakespeare",
    created_at: "2026-07-01 10:00:00",
    updated_at: "2026-07-01 10:00:00",
    ...overrides,
  } as import("../types").Experiment;
}

describe("ExperimentBrowser", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders nothing while loading or when there are no experiments", async () => {
    vi.spyOn(api, "listExperiments").mockResolvedValue([]);

    const { container } = render(<ExperimentBrowser onSelect={() => {}} />);

    await waitFor(() => expect(api.listExperiments).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("lists existing experiments, most recently updated first", async () => {
    vi.spyOn(api, "listExperiments").mockResolvedValue([
      makeExperiment({ id: 1, name: "Older", updated_at: "2026-07-01 10:00:00" }),
      makeExperiment({ id: 2, name: "Newer", updated_at: "2026-07-05 10:00:00" }),
    ]);

    render(<ExperimentBrowser onSelect={() => {}} />);

    await waitFor(() => expect(screen.getByText("Newer")).toBeInTheDocument());
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).toHaveTextContent("Newer");
    expect(buttons[1]).toHaveTextContent("Older");
  });

  it("calls onSelect with the experiment id and config when clicked", async () => {
    const exp = makeExperiment();
    vi.spyOn(api, "listExperiments").mockResolvedValue([exp]);
    const onSelect = vi.fn();

    render(<ExperimentBrowser onSelect={onSelect} />);

    await waitFor(() => expect(screen.getByText("Tiny Shakespeare Run")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Tiny Shakespeare Run"));

    expect(onSelect).toHaveBeenCalledWith(3, exp.config);
  });
});
