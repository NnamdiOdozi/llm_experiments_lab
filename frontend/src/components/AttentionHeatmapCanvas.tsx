import { useState, useEffect, useRef } from "react";
import { AttentionFull } from "../types";

/**
 * Pure functions for colour mapping and label stride — testable without canvas.
 */

/** Map attention weight [0,1] to RGB, with optional gamma correction. */
export function weightToColor(weight: number, gamma: number = 0.5): string {
  // Clamp weight to [0, 1] and apply gamma
  const clamped = Math.max(0, Math.min(1, weight));
  const corrected = Math.pow(clamped, gamma);
  // Sequential ramp: blue-to-red gradient (HSL → RGB)
  // At 0: blue (240°), at 1: red (0°)
  const hue = 240 * (1 - corrected);
  const saturation = 100;
  const lightness = 50;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/** Compute label stride based on sequence length and available pixel width. */
export function computeLabelStride(
  sequenceLength: number,
  pixelWidth: number,
  minPixelsPerLabel: number = 40,
): number {
  if (sequenceLength === 0 || pixelWidth === 0) return 1;
  const labelsPerPixel = sequenceLength / pixelWidth;
  // stride = max(1, ceil(labelsPerPixel * minPixelsPerLabel))
  const stride = Math.max(1, Math.ceil(labelsPerPixel * minPixelsPerLabel));
  // Prefer powers of 2: 1, 2, 4, 8, 16, ...
  return Math.pow(2, Math.ceil(Math.log2(stride)));
}

// Fixed drawing height for the canvas (px). The container height auto-fits
// header + canvas, so the canvas must NOT be sized from the container.
const CANVAS_HEIGHT = 340;
const LEFT_MARGIN = 60;
const TOP_MARGIN = 72;
const RIGHT_MARGIN = 84;
const BOTTOM_MARGIN = 24;

function heatmapLayout(width: number, height: number) {
  return {
    leftMargin: LEFT_MARGIN,
    topMargin: TOP_MARGIN,
    squareSize: Math.max(0, Math.min(
      width - LEFT_MARGIN - RIGHT_MARGIN,
      height - TOP_MARGIN - BOTTOM_MARGIN,
    )),
  };
}

interface AttentionHeatmapCanvasProps {
  attentionFull: AttentionFull | null;
  blockNum: number | null;
  head: number | null;
}

export default function AttentionHeatmapCanvas({
  attentionFull,
  blockNum,
  head,
}: AttentionHeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    queryIdx: number;
    keyIdx: number;
    queryToken: string;
    keyToken: string;
    weight: number;
  } | null>(null);

  // Handle window/container resize (guarded for jsdom test environments)
  useEffect(() => {
    if (typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(() => {
      if (canvasRef.current && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const pixelRatio = window.devicePixelRatio || 1;
        canvasRef.current.width = Math.max(100, Math.round(rect.width * pixelRatio));
        // Height is a fixed constant (matches the canvas CSS height) — NOT the
        // container's height. Deriving it from the container (which grows to
        // fit header + canvas) made the canvas taller than its box, so it
        // overflowed and covered the Q/K/V line below it (user report,
        // 2026-08-06).
        canvasRef.current.height = Math.round(CANVAS_HEIGHT * pixelRatio);
        render();
      }
    });

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Render on data change
  useEffect(() => {
    render();
  }, [attentionFull]);

  const render = () => {
    const canvas = canvasRef.current;
    if (!canvas || !attentionFull?.available || !attentionFull.weights || !attentionFull.token_labels) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      // jsdom or environments without 2D context
      return;
    }

    const weights = attentionFull.weights;
    const labels = attentionFull.token_labels;
    const sequenceLength = labels.length;

    if (sequenceLength === 0 || weights.length === 0) {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // Draw in CSS-pixel coordinates while keeping a device-pixel backing
    // store, so labels stay sharp under browser zoom and on HiDPI screens.
    const pixelRatio = window.devicePixelRatio || 1;
    const logicalWidth = canvas.width / pixelRatio;
    const logicalHeight = canvas.height / pixelRatio;
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const { leftMargin, topMargin, squareSize } = heatmapLayout(logicalWidth, logicalHeight);
    const cellSize = squareSize / sequenceLength;

    // Clear background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, logicalWidth, logicalHeight);

    // Draw heatmap cells
    for (let i = 0; i < sequenceLength; i++) {
      for (let j = 0; j < sequenceLength; j++) {
        const x = leftMargin + j * cellSize;
        const y = topMargin + i * cellSize;

        // Causal masking: j > i is masked
        if (j > i) {
          // Distinct pattern for masked cells (diagonal stripes)
          ctx.fillStyle = "#3a3a3a";
          ctx.fillRect(x, y, cellSize, cellSize);
          ctx.strokeStyle = "#2a2a2a";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cellSize, cellSize);

          // Light diagonal stripe pattern
          ctx.strokeStyle = "#555555";
          ctx.lineWidth = 0.25;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + cellSize, y + cellSize);
          ctx.stroke();
        } else {
          // Real attention weight
          const weight = weights[i]?.[j] ?? 0;
          ctx.fillStyle = weightToColor(weight);
          ctx.fillRect(x, y, cellSize, cellSize);

          // Cell border
          ctx.strokeStyle = "#2a2a2a";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cellSize, cellSize);
        }
      }
    }

    // Draw row labels (query positions, left side)
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "600 11px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < sequenceLength; i++) {
      const y = topMargin + i * cellSize + cellSize / 2;
      ctx.fillText(labels[i] || "?", leftMargin - 10, y);
    }

    // Draw column labels (key positions, top)
    ctx.fillStyle = "#f8fafc";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    // Character/token labels only need a compact slot. The old 40px minimum
    // skipped alternating keys even for a short 8–10 token sequence.
    const stride = computeLabelStride(sequenceLength, squareSize, 16);
    for (let j = 0; j < sequenceLength; j += stride) {
      const x = leftMargin + j * cellSize + cellSize / 2;
      ctx.fillText(labels[j] || "?", x, topMargin - 23);
    }

    // Draw axis labels
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 12px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Key (attended to) →", leftMargin + squareSize / 2, 12);

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "600 11px monospace";
    ctx.save();
    ctx.translate(15, topMargin + squareSize / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Query (attending from) →", 0, 0);
    ctx.restore();

    // Draw legend
    const legendX = leftMargin + squareSize + 14;
    const legendY = topMargin + 10;
    const legendHeight = 150;
    const legendWidth = 16;

    // Gradient from 0 to 1
    const gradientSegments = 100;
    for (let seg = 0; seg < gradientSegments; seg++) {
      // Top of the legend = high attention (matches the "High"/"Low" labels
      // drawn top/bottom below), so invert: seg 0 (top) -> weight 1.
      const weight = 1 - seg / gradientSegments;
      ctx.fillStyle = weightToColor(weight);
      ctx.fillRect(
        legendX,
        legendY + (seg / gradientSegments) * legendHeight,
        legendWidth,
        legendHeight / gradientSegments + 1,
      );
    }

    // Legend border
    ctx.strokeStyle = "#666666";
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Legend labels
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("High", legendX + legendWidth + 5, legendY);
    ctx.fillText("Low", legendX + legendWidth + 5, legendY + legendHeight);

    // Legend title
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Attention", legendX + legendWidth / 2, legendY - 16);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !attentionFull?.available || !attentionFull.weights || !attentionFull.token_labels) {
      setTooltip(null);
      return;
    }

    const weights = attentionFull.weights;
    const labels = attentionFull.token_labels;
    const sequenceLength = labels.length;

    const rect = canvas.getBoundingClientRect();
    const { leftMargin, topMargin, squareSize } = heatmapLayout(rect.width, rect.height);
    const cellSize = squareSize / sequenceLength;
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;

    // Check if within heatmap bounds
    if (
      offsetX < leftMargin ||
      offsetX > leftMargin + squareSize ||
      offsetY < topMargin ||
      offsetY > topMargin + squareSize
    ) {
      setTooltip(null);
      return;
    }

    const queryIdx = Math.floor((offsetY - topMargin) / cellSize);
    const keyIdx = Math.floor((offsetX - leftMargin) / cellSize);

    if (queryIdx >= 0 && queryIdx < sequenceLength && keyIdx >= 0 && keyIdx < sequenceLength) {
      const weight = weights[queryIdx]?.[keyIdx] ?? 0;
      setTooltip({
        x: offsetX,
        y: offsetY,
        queryIdx,
        keyIdx,
        queryToken: labels[queryIdx] || "?",
        keyToken: labels[keyIdx] || "?",
        weight,
      });
    } else {
      setTooltip(null);
    }
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  if (!attentionFull?.available) {
    return (
      <div style={{ fontSize: 15, color: "var(--text-dim)" }}>
        {attentionFull?.reason ? `Not captured: ${attentionFull.reason}` : "Not captured"}
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%" }}>
      <div style={{ fontSize: 14, color: "var(--accent)", marginBottom: 8 }}>
        Layer {blockNum != null ? blockNum + 1 : "?"}, Head {head != null ? head + 1 : "?"}
        {attentionFull.total_positions && ` — ${attentionFull.total_positions} positions`}
      </div>
      <canvas
        ref={canvasRef}
        width={100}
        height={100}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          width: "100%",
          height: CANVAS_HEIGHT,
          display: "block",
          backgroundColor: "#1a1a1a",
          borderRadius: 4,
          cursor: "crosshair",
        }}
      />
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: `${tooltip.x + 10}px`,
            top: `${tooltip.y + 10}px`,
            background: "rgba(0, 0, 0, 0.9)",
            color: "#fff",
            padding: "8px 12px",
            borderRadius: 4,
            fontSize: 12,
            fontFamily: "monospace",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 1000,
            border: "1px solid #444",
          }}
        >
          <div>Query: "{tooltip.queryToken}" (pos {tooltip.queryIdx})</div>
          <div>Key: "{tooltip.keyToken}" (pos {tooltip.keyIdx})</div>
          <div>Weight: {tooltip.weight.toFixed(4)}</div>
        </div>
      )}
    </div>
  );
}
