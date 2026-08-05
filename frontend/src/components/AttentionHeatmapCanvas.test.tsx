import { describe, it, expect } from "vitest";
import { weightToColor, computeLabelStride } from "./AttentionHeatmapCanvas";

// The canvas rendering itself can't be asserted in jsdom (no 2D context), so
// the colour + label-stride logic is extracted as pure functions and tested
// directly here.

describe("weightToColor", () => {
  it("returns an hsl string", () => {
    expect(weightToColor(0.5)).toMatch(/^hsl\(/);
  });

  it("maps low weights toward blue (hue ~240) and high toward red (hue ~0)", () => {
    const low = weightToColor(0); // hue = 240*(1-0^gamma) = 240
    const high = weightToColor(1); // hue = 240*(1-1) = 0
    expect(low).toContain("hsl(240");
    expect(high).toContain("hsl(0");
  });

  it("clamps out-of-range weights", () => {
    expect(weightToColor(-5)).toContain("hsl(240"); // clamped to 0
    expect(weightToColor(5)).toContain("hsl(0");     // clamped to 1
  });

  it("gamma < 1 lifts small values (sqrt default makes 0.25 read brighter than linear)", () => {
    // gamma 0.5: corrected = sqrt(0.25) = 0.5 → hue = 120; linear would be 180.
    expect(weightToColor(0.25)).toContain("hsl(120");
  });
});

describe("computeLabelStride", () => {
  it("is 1 when there is plenty of room per label", () => {
    expect(computeLabelStride(8, 800)).toBe(1);
  });

  it("grows (power of two) as the sequence crowds the available width", () => {
    const stride = computeLabelStride(256, 300);
    expect(stride).toBeGreaterThan(1);
    // Always a power of two.
    expect(Math.log2(stride) % 1).toBe(0);
  });

  it("guards against zero inputs", () => {
    expect(computeLabelStride(0, 100)).toBe(1);
    expect(computeLabelStride(100, 0)).toBe(1);
  });
});
