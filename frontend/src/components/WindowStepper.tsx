
export interface WindowStepperProps {
  windowStart: number;
  windowSize: number;
  totalPositions: number;
  offset: number;
  onOffsetChange: (offset: number) => void;
}

/**
 * Reusable stepper UI for windowed position views. Renders:
 * ◀ Earlier / "Positions X–Y of Z" / Later ▶
 *
 * With stride=1 smooth stepping (one position at a time), disabled state
 * when at boundaries, and returns null if window covers entire sequence.
 *
 * Used by both NodeWindowStepper (for generic node vectors) and
 * AttentionHeatmap (for attention weights window). Same math, same UI,
 * differs only in source of windowStart/windowSize/totalPositions
 * (and hide condition, which caller handles).
 *
 * See docs/DESIGN_DECISIONS.md for background on windowed views.
 */
export function WindowStepper({
  windowStart,
  windowSize,
  totalPositions,
  offset,
  onOffsetChange,
}: WindowStepperProps) {
  const maxOffset = Math.max(0, totalPositions - windowSize);
  const canShiftEarlier = offset < maxOffset;
  const canShiftLater = offset > 0;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 11 }}>
      <button
        onClick={() => onOffsetChange(Math.min(maxOffset, offset + 1))}
        disabled={!canShiftEarlier}
        title="Shift window one position earlier"
        style={{ fontSize: 11, padding: "2px 6px" }}
      >
        ◀ Earlier
      </button>
      <span style={{ color: "var(--text-dim)" }}>
        Positions {windowStart + 1}–{windowStart + windowSize} of {totalPositions}
      </span>
      <button
        onClick={() => onOffsetChange(Math.max(0, offset - 1))}
        disabled={!canShiftLater}
        title="Shift window one position later"
        style={{ fontSize: 11, padding: "2px 6px" }}
      >
        Later ▶
      </button>
    </div>
  );
}
