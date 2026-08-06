import { useState, useEffect, useRef, useLayoutEffect, type Ref } from "react";
import { ArchitectureManifest, ArchitectureNode, ExperimentConfig } from "../types";
import { fetchArchitecture, previewArchitecture } from "../hooks/useApi";
import "./ArchSchematic.css";

// Debounce for /architecture/preview while the user is still editing
// ConfigPanel (e.g. dragging a layer-count slider) — avoids one request per
// keystroke/tick. Direct user request, 2026-07-15.
const PREVIEW_DEBOUNCE_MS = 400;

interface Props {
  runId?: number | null;
  // Live experiment config, used only when there's no runId yet — lets the
  // diagram render before Start is clicked and update as structural fields
  // (n_layer/n_head/n_embd/pos_encoding) change. Ignored once a run exists;
  // the run's frozen config_snapshot takes over via runId. See
  // docs/DESIGN_DECISIONS.md.
  config?: ExperimentConfig | null;
  onNodeClick?: (nodeId: string, node: ArchitectureNode) => void;
  selectedNodeId?: string | null;
}

// Measured geometry for a detail box (relative to flow container).
interface BoxGeometry {
  left: number;
  right: number;
  centerX: number;
  top?: number;
  bottom?: number;
  height?: number;
}

interface BoxProps {
  label: string;
  fullLabel?: string;
  kind: string;
  elementRef?: Ref<HTMLDivElement>;
  isSelected?: boolean;
  isExpanded?: boolean;
  onClick?: () => void;
  // Renders 3 small internal stripes to signal "multiple experts inside"
  // without making it look like 3 separately-clickable boxes — previously
  // this was 3 literal NodeBoxes all wired to the same node id, which read
  // as misleading (implies per-expert data that doesn't exist; MoE is
  // captured as one opaque node). Direct user report, 2026-07-15. See
  // docs/DESIGN_DECISIONS.md.
  segmented?: boolean;
}

// MoE and dense MLP are the same color/kind of thing at a glance — MoE is
// just "a few of these instead of one" (per user 2026-07-13), not a
// structurally different-looking component. Kept as one shared entry.
const COLOR_MAP: Record<string, { bg: string; border: string }> = {
  embedding: { bg: "#dbeafe", border: "#0284c7" },
  layernorm: { bg: "#fef08a", border: "#ca8a04" },
  attention: { bg: "#fdd2f3", border: "#be185d" },
  mlp: { bg: "#cffafe", border: "#0891b2" },
  moe: { bg: "#cffafe", border: "#0891b2" },
  rnn: { bg: "#fecaca", border: "#dc2626" },
  dropout: { bg: "#e5e7eb", border: "#9ca3af" },
  lm_head: { bg: "#f3e8ff", border: "#a855f7" },
  transformer_block_group: { bg: "#e0e7ff", border: "#4f46e5" },
};

// Keep the architecture manifest's full technical labels intact for the
// Inspector and data contract. These are display-only abbreviations for the
// compact schematic, where the surrounding flow already supplies context.
const DISPLAY_LABELS: Record<string, string> = {
  "Token + Positional Embedding": "T + P Embedding",
  "Final LayerNorm": "Final\nLayerNorm",
  "LayerNorm (pre-attention)": "LayerNorm",
  "LayerNorm (pre-MLP)": "LayerNorm",
  "LayerNorm (pre-MLP/MoE)": "LayerNorm",
  "Causal Self-Attention": "Causal S. Attention",
  "Feed-Forward (dense)": "Feed Forward",
};

type ArchitectureVisualVariant = "minimal" | "technical" | "educational";

const ARCHITECTURE_VISUAL_VARIANTS = new Set<ArchitectureVisualVariant>([
  "minimal",
  "technical",
  "educational",
]);

const NODE_KIND_LABELS: Record<string, string> = {
  layernorm: "LN",
  attention: "ATTN",
  mlp: "FFN",
  moe: "MOE",
  embedding: "EMB",
  lm_head: "HEAD",
  transformer_block_group: "BLOCK",
};

const DETAIL_STAGE_CAPTIONS = [
  "01 · Normalization",
  "02 · Attention",
  "03 · Normalization",
  "04 · Feed-forward",
] as const;

function architectureVisualVariant(): ArchitectureVisualVariant {
  const requested = new URLSearchParams(window.location.search).get("archVariant");
  return ARCHITECTURE_VISUAL_VARIANTS.has(requested as ArchitectureVisualVariant)
    ? requested as ArchitectureVisualVariant
    : "minimal";
}

function displayLabel(label: string) {
  return DISPLAY_LABELS[label] ?? label;
}

// Remote serverless workers can temporarily run an older manifest producer
// than this frontend. Identify the shared Transformer/MoE block group by its
// stable contract id as well as the current kind, rather than relying on the
// human-readable MoE label or a single producer version.
function isTransformerBlockGroup(node: ArchitectureNode) {
  return node.id === "block" || node.kind === "transformer_block_group";
}

function NodeBox({ label, fullLabel, kind, elementRef, isSelected, isExpanded, onClick, segmented }: BoxProps) {
  const colors = COLOR_MAP[kind] || { bg: "#e5e7eb", border: "#6b7280" };

  return (
    <div
      ref={elementRef}
      data-kind={kind}
      className={`arch-node${onClick ? " arch-node--interactive" : ""}${isSelected ? " arch-node--selected" : ""}${isExpanded ? " arch-node--expanded" : ""}`}
      onClick={onClick}
      onKeyDown={onClick ? (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick();
        }
      } : undefined}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={onClick ? (fullLabel || label).replace(/\n/g, " ") : undefined}
      aria-pressed={onClick ? Boolean(isSelected) : undefined}
      title={fullLabel && fullLabel !== label ? fullLabel : undefined}
      style={{
        backgroundColor: colors.bg,
        border: `2px solid ${colors.border}`,
        borderColor: isSelected ? "#ff6b6b" : colors.border,
        boxShadow: isSelected ? "0 0 8px rgba(255, 107, 107, 0.5)" : "none",
      }}
    >
      {segmented && (
        <div className="arch-node__segments" aria-hidden="true">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="arch-node__segment"
              style={{ backgroundColor: colors.border }}
            />
          ))}
        </div>
      )}
      {NODE_KIND_LABELS[kind] && (
        <span className="arch-node__kind" aria-hidden="true">{NODE_KIND_LABELS[kind]}</span>
      )}
      <span className="arch-node__label">{label}</span>
    </div>
  );
}

function Arrow({
  variant,
  elementRef,
  muted,
}: {
  variant?: "entry" | "exit";
  elementRef?: Ref<HTMLDivElement>;
  // True when the residual overlay is drawing its own connector through this
  // exact gap (see ResidualStreamOverlay) — the div stays in the DOM so its
  // ref still reports real geometry, it just doesn't paint.
  muted?: boolean;
}) {
  return (
    <div
      ref={elementRef}
      className={`arch-arrow${variant ? ` arch-arrow--${variant}` : ""}${muted ? " arch-arrow--muted" : ""}`}
      aria-hidden="true"
    />
  );
}

// Leave a deliberate inset around both architecture canvases. Fitting exactly
// edge-to-edge is vulnerable to browser zoom/sub-pixel rounding and made the
// rightmost Output / Feed Forward / Experts nodes look truncated in a narrow
// dashboard column. Direct user request, 2026-08-06.
const ARCHITECTURE_FIT_RATIO = 0.9;

// Scales its (natural-width) children down uniformly to 90% of the available
// width and centers them — used for the top pipeline so it never touches the
// clipping edge. Measures in layout px, unaffected by the global app zoom.
function FitScale({ children }: { children: React.ReactNode }) {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [leftInset, setLeftInset] = useState(0);
  const [height, setHeight] = useState<number | undefined>(undefined);
  useLayoutEffect(() => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;
    const measure = () => {
      const natural = Math.max(inner.offsetWidth, inner.scrollWidth);
      const avail = outer.clientWidth;
      const usable = avail * ARCHITECTURE_FIT_RATIO;
      const s = natural > 0 && usable > 0 ? Math.min(1, usable / natural) : 1;
      setScale(s);
      setLeftInset(Math.max(0, (avail - natural * s) / 2));
      setHeight(inner.offsetHeight * s);
    };
    measure();
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => measure());
      observer.observe(outer);
      observer.observe(inner);
    }
    window.addEventListener("resize", measure);
    return () => {
      if (observer) observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  });
  return (
    <div ref={outerRef} style={{ width: "100%", overflow: "hidden", height }}>
      <div
        ref={innerRef}
        style={{
          width: "max-content",
          transformOrigin: "top left",
          transform: `translateX(${leftInset}px) scale(${scale})`,
        }}
      >
        {children}
      </div>
    </div>
  );
}

// Centralized drawing dimensions keep the residual diagram internally
// consistent and make future density changes a one-place edit.
const RESIDUAL_DIAGRAM = {
  streamY: 26,
  branchRadius: 5,
  operatorRadius: 14,
  attentionAddOffset: -6,
  outputAddOffset: -12,
  secondTapOffset: 8,
  flowArrowOffset: 18,
  flowArrowHalfHeight: 4,
  // Size of the arrowhead drawn where a main-flow connector (see
  // mainFlowConnector below) arrives at the next box — thin tier for an
  // ordinary hop, strong tier for the block-boundary hop.
  connectorArrowThin: 7,
  connectorArrowStrong: 9,
} as const;

/**
 * Residual stream overlay
 *
 * A pre-norm transformer has one evolving residual stream. The first input is
 * split into the attention sublayer and its bypass. That first bypass ends at
 * the attention addition; a distinct tap just after the addition starts the
 * MLP bypass. The visible break between spans makes the updated-stream handoff
 * explicit instead of implying one unchanged rail across both sublayers.
 */
function ResidualStreamOverlay({
  geometry,
  entryArrow,
  exitArrow,
  containerHeight = 100,
}: {
  geometry: Map<number, BoxGeometry>;
  entryArrow: BoxGeometry | null;
  exitArrow: BoxGeometry | null;
  containerHeight?: number;
}) {
  // Verify both required boxes exist: attention (idx 1) and mlp/moe (idx 3).
  const ln1 = geometry.get(0);
  const attention = geometry.get(1);
  const ln2 = geometry.get(2);
  const mlpOrMoe = geometry.get(3);

  if (!ln1 || !attention || !ln2 || !mlpOrMoe || !entryArrow || !exitArrow) {
    // Missing required boxes or arrows — don't render brackets (graceful degrade).
    return null;
  }

  // Use the measured node row rather than a fixed vertical center, so the
  // operators stay aligned if node height or detail spacing changes.
  let centerlineY = containerHeight / 2; // fallback if height data unavailable
  if (ln1.top !== undefined && ln1.height !== undefined) {
    centerlineY = ln1.top + ln1.height / 2;
  }
  // Start at the outer edge of the entry connector: the residual stream is
  // present from the block's very first input, before LayerNorm is applied.
  const inputBranchX = entryArrow.centerX;
  const attentionAddX = (attention.right + ln2.left) / 2
    + RESIDUAL_DIAGRAM.attentionAddOffset;
  const secondBranchX = attentionAddX
    + RESIDUAL_DIAGRAM.operatorRadius
    + RESIDUAL_DIAGRAM.secondTapOffset;
  const outputAddX = exitArrow.centerX + RESIDUAL_DIAGRAM.outputAddOffset;

  const streamPath = (startX: number, endX: number) =>
    `M ${startX} ${centerlineY} V ${RESIDUAL_DIAGRAM.streamY} H ${endX} V ${centerlineY}`;

  const flowChevron = (endX: number) => {
    const x = endX - RESIDUAL_DIAGRAM.flowArrowOffset;
    return `M ${x - 4} ${RESIDUAL_DIAGRAM.streamY - RESIDUAL_DIAGRAM.flowArrowHalfHeight} L ${x} ${RESIDUAL_DIAGRAM.streamY} L ${x - 4} ${RESIDUAL_DIAGRAM.streamY + RESIDUAL_DIAGRAM.flowArrowHalfHeight}`;
  };

  // Draws the main-flow connector THROUGH an operator ring — the plain
  // between-box arrow at this gap is muted (see Arrow's `muted` prop) so
  // this is the only thing painting there. Same shaft-then-arrowhead shape
  // as the plain/entry-exit arrows it replaces, just anchored to measured
  // box edges instead of a fixed flex-basis width.
  const mainFlowConnector = (startX: number, endX: number, strong: boolean) => {
    const size = strong ? RESIDUAL_DIAGRAM.connectorArrowStrong : RESIDUAL_DIAGRAM.connectorArrowThin;
    const shaftClass = strong ? "arch-residual-overlay__connector-strong" : "arch-residual-overlay__connector";
    const headClass = strong
      ? "arch-residual-overlay__connector-arrowhead--strong"
      : "arch-residual-overlay__connector-arrowhead";
    return (
      <g data-testid="residual-connector">
        <line className={shaftClass} x1={startX} y1={centerlineY} x2={endX - size} y2={centerlineY} />
        <path
          className={headClass}
          d={`M ${endX - size} ${centerlineY - size} L ${endX} ${centerlineY} L ${endX - size} ${centerlineY + size}`}
        />
      </g>
    );
  };

  const additionOperator = (x: number) => (
    <g data-testid="residual-plus">
      <circle className="arch-residual-overlay__operator" cx={x} cy={centerlineY} r={RESIDUAL_DIAGRAM.operatorRadius} />
      <text
        className="arch-residual-overlay__plus"
        x={x}
        y={centerlineY}
        textAnchor="middle"
        dominantBaseline="central"
      >
        +
      </text>
    </g>
  );

  return (
    <svg
      className="arch-residual-overlay"
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "visible",
      }}
      aria-hidden="true"
    >
      <text
        className="arch-residual-overlay__label"
        x={inputBranchX + 12}
        y={RESIDUAL_DIAGRAM.streamY - 10}
      >
        RESIDUAL STREAM
      </text>

      {/* First bypass: original input skips LayerNorm + attention. */}
      <path
        className="arch-residual-overlay__stream"
        d={streamPath(inputBranchX, attentionAddX)}
        data-testid="residual-arc"
      />

      {/* The updated stream is tapped again for LayerNorm + feed-forward. */}
      <path
        className="arch-residual-overlay__stream"
        d={streamPath(secondBranchX, outputAddX)}
        data-testid="residual-arc"
      />

      {/* Direction cues make the upper rail read as a stream, not decoration. */}
      <path
        className="arch-residual-overlay__chevron"
        d={flowChevron(attentionAddX)}
      />
      <path
        className="arch-residual-overlay__chevron"
        d={flowChevron(outputAddX)}
      />

      {/* Two distinct divergence points: the original block input and the
          updated residual stream just after the attention addition. */}
      <circle
        className="arch-residual-overlay__branch"
        cx={inputBranchX}
        cy={centerlineY}
        r={RESIDUAL_DIAGRAM.branchRadius}
        data-testid="residual-branch-dot"
      />
      <circle
        className="arch-residual-overlay__branch"
        cx={secondBranchX}
        cy={centerlineY}
        r={RESIDUAL_DIAGRAM.branchRadius}
        data-testid="residual-branch-dot"
        data-residual-role="second-tap"
      />

      {/* Main-flow connectors, painted before (under) the operator rings so
          each "+" reads as an inline junction on one continuous line rather
          than a badge covering a separate arrow. */}
      {mainFlowConnector(attention.right, ln2.left, false)}
      {mainFlowConnector(mlpOrMoe.right, exitArrow.right, true)}

      {additionOperator(attentionAddX)}
      {additionOperator(outputAddX)}
    </svg>
  );
}

export default function ArchSchematic({ runId, config, onNodeClick, selectedNodeId }: Props) {
  const visualVariant = architectureVisualVariant();
  const [manifest, setManifest] = useState<ArchitectureManifest | null>(null);
  const [selectedBlockIdx, setSelectedBlockIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  // Tracks whether at least one fetch attempt has failed — distinguishes
  // "still loading" from "worker's cold-starting, retrying" in the UI.
  const [retrying, setRetrying] = useState(false);

  // Residual connection visualization state — measures detail box positions
  // for rendering skip brackets. Keyed by child index (0=ln1, 1=attention, 2=ln2, 3=mlp/moe).
  // Also tracks entry/exit arrow positions for bracket anchoring.
  const [residualGeometry, setResidualGeometry] = useState<Map<number, BoxGeometry>>(new Map());
  const [entryArrowGeometry, setEntryArrowGeometry] = useState<BoxGeometry | null>(null);
  const [exitArrowGeometry, setExitArrowGeometry] = useState<BoxGeometry | null>(null);
  const detailBoxRefs = useRef<Map<number, HTMLElement>>(new Map());
  const entryArrowRef = useRef<HTMLDivElement>(null);
  const exitArrowRef = useRef<HTMLDivElement>(null);
  const detailFlowContainerRef = useRef<HTMLDivElement>(null);
  // Fit-to-width: the inset lays out at its natural width and is CSS-scaled down
  // to fit its dashboard column (computed in the geometry effect below).
  const [fitScale, setFitScale] = useState(1);
  const [fitInset, setFitInset] = useState(0);
  const [fitNaturalWidth, setFitNaturalWidth] = useState<number | null>(null);
  const [fitHeight, setFitHeight] = useState<number | null>(null);

  // Serverless GPU workers cold-start on first use — a real, observed 7.5
  // minute wake-up (session log, 2026-07-13 21:32-21:40). A run always gets
  // created right as that cold start begins, so this component's first
  // fetch reliably lands in the middle of it and 502s. Previously a single
  // fetch-and-give-up: once that first attempt failed, the panel stayed
  // permanently blank even after the worker woke up seconds later, until a
  // full page reload. Now retries on a timer until it succeeds or runId
  // changes. See docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const attempt = (isRetry: boolean) => {
      if (cancelled) return;
      setLoading(!isRetry);
      setRetrying(isRetry);
      fetchArchitecture(runId)
        .then((m) => {
          if (cancelled) return;
          setManifest(m);
          setLoading(false);
          setRetrying(false);
        })
        .catch(() => {
          if (cancelled) return;
          setManifest(null);
          setLoading(false);
          timer = setTimeout(() => attempt(true), 5000);
        });
    };
    attempt(false);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  // Preview mode — no run yet, just the experiment's (possibly still being
  // edited) config. Debounced so dragging a slider in ConfigPanel doesn't
  // fire one request per tick. Only active while there's no runId; once a
  // run exists the effect above takes over and this one goes quiet — the
  // run's frozen config_snapshot is what should drive the diagram from
  // then on, not further live edits. Direct user request, 2026-07-15. See
  // docs/DESIGN_DECISIONS.md.
  useEffect(() => {
    if (runId || !config) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      previewArchitecture(config)
        .then((m) => {
          if (!cancelled) setManifest(m);
        })
        .catch(() => {
          if (!cancelled) setManifest(null);
        });
    }, PREVIEW_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId, config]);

  // Residual connection visualization — measure detail box positions and entry/exit arrows
  // whenever manifest or selected block changes. Uses ResizeObserver (if available) to
  // track width changes. Does NOT run in jsdom (tests will get zero rects,
  // which is fine — the SVG elements still render). See docs/DESIGN_DECISIONS.md.
  useLayoutEffect(() => {
    const container = detailFlowContainerRef.current;
    const refs = detailBoxRefs.current;
    const entryArrow = entryArrowRef.current;
    const exitArrow = exitArrowRef.current;
    if (!container || refs.size === 0) {
      setResidualGeometry(new Map());
      setEntryArrowGeometry(null);
      setExitArrowGeometry(null);
      return;
    }

    const measureGeometry = () => {
      // Measure every box in the FLOW'S coordinate system. `offsetLeft` cannot
      // be used directly here: each node's positioned `.arch-stage` becomes
      // its nearest offset parent, which makes all four boxes appear to begin
      // close to x=0 and collapses the first residual span. DOM rects include
      // both the local fit transform and the dashboard's global CSS zoom, so
      // divide by the observed scale to recover the natural SVG coordinates.
      const containerRect = container.getBoundingClientRect();
      const scaleX = container.offsetWidth > 0 && containerRect.width > 0
        ? containerRect.width / container.offsetWidth
        : 1;
      const scaleY = container.offsetHeight > 0 && containerRect.height > 0
        ? containerRect.height / container.offsetHeight
        : scaleX;
      const relativeGeometry = (el: HTMLElement): BoxGeometry => {
        const rect = el.getBoundingClientRect();
        const left = (rect.left - containerRect.left) / scaleX;
        const top = (rect.top - containerRect.top) / scaleY;
        const width = rect.width / scaleX;
        const height = rect.height / scaleY;
        return {
          left,
          right: left + width,
          centerX: left + width / 2,
          top,
          bottom: top + height,
          height,
        };
      };

      const geometry = new Map<number, BoxGeometry>();
      refs.forEach((el, idx) => {
        geometry.set(idx, relativeGeometry(el));
      });
      setResidualGeometry(geometry);

      if (entryArrow) {
        setEntryArrowGeometry(relativeGeometry(entryArrow));
      } else {
        setEntryArrowGeometry({ left: 0, right: 0, centerX: 0 });
      }

      if (exitArrow) {
        setExitArrowGeometry(relativeGeometry(exitArrow));
      } else {
        setExitArrowGeometry({
          left: container.offsetWidth,
          right: container.offsetWidth,
          centerX: container.offsetWidth,
        });
      }

      // Fit-to-width: scale the natural-width flow down to its clipping parent
      // (the dashboard column) so the whole inset is visible without scrolling
      // and shrinks uniformly with the column. offsetWidth/scrollHeight are
      // layout values (unaffected by the transform we set), so this is stable.
      const fitParent = container.parentElement;
      const naturalWidth = Math.max(container.offsetWidth, container.scrollWidth);
      const avail = fitParent ? fitParent.clientWidth : naturalWidth;
      const usable = avail * ARCHITECTURE_FIT_RATIO;
      const scale = naturalWidth > 0 && usable > 0 ? Math.min(1, usable / naturalWidth) : 1;
      setFitNaturalWidth(naturalWidth);
      setFitScale(scale);
      setFitInset(Math.max(0, (avail - naturalWidth * scale) / 2));
      // Reserve only the SCALED height so nothing empty is left below (transform
      // doesn't shrink the layout box on its own).
      setFitHeight(container.offsetHeight * scale);
    };

    // Measure immediately and on resize.
    measureGeometry();
    const handleResize = () => measureGeometry();
    window.addEventListener("resize", handleResize);

    // ResizeObserver is not available in jsdom; gracefully skip if missing.
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => measureGeometry());
      // Observe the clipping parent (the dashboard column), whose width changes
      // when a side-pane opens — the flow's own natural width does not, so
      // observing it wouldn't catch a column resize.
      if (container.parentElement) observer.observe(container.parentElement);
      observer.observe(container);
    }

    return () => {
      if (observer) observer.disconnect();
      window.removeEventListener("resize", handleResize);
    };
  }, [manifest, selectedBlockIdx]);

  if (loading) {
    return <div className="panel"><h3>Architecture</h3><p>Loading...</p></div>;
  }

  if (retrying && !manifest) {
    return (
      <div className="panel">
        <h3>Architecture</h3>
        <p style={{ fontSize: 12, color: "var(--text-dim)" }}>
          Waking up remote worker — serverless cold starts can take several minutes. Retrying...
        </p>
      </div>
    );
  }

  const nodes = manifest ? manifest.nodes : [];

  return (
    <div
      className={`panel arch-schematic arch-schematic--${visualVariant}`}
      id="arch-schematic"
      data-visual-variant={visualVariant}
    >
      <h3>Architecture</h3>

      {/* Horizontal pipeline diagram — wrapped in FitScale so it scales down to
          fit its column instead of clipping/scrolling (e.g. LM Head cut off at
          the right). 2026-08-06 user report. */}
      <FitScale>
      <div className="arch-pipeline">
        {/* Input/Output bookends — direct user request, 2026-07-15: no
            indication anywhere of where data enters/exits the model.
            "io" is deliberately not in COLOR_MAP, so NodeBox falls back to
            its neutral gray — these are boundary labels, not real
            computational nodes, and shouldn't be color-coded like one.
            Shared by every template (transformer/MoE/RNN all render
            through this same nodes.map). See docs/DESIGN_DECISIONS.md. */}
        {nodes.length > 0 && (
          <div className="arch-flow-item arch-flow-item--io">
            <div className="arch-stage">
              <NodeBox label="Input" kind="io" />
            </div>
          </div>
        )}
        {nodes.length > 0 && <Arrow />}
        {nodes.map((node) => {
          if (isTransformerBlockGroup(node) && node.children && node.repeat_count) {
            return (
              <div key={node.id} className="arch-flow-item arch-flow-item--block">
                <div className="arch-stage">
                  <NodeBox
                    label={`Transformer\nBlock ${selectedBlockIdx + 1} of ${node.repeat_count}`}
                    fullLabel={node.label}
                    kind="transformer_block_group"
                    isExpanded
                    onClick={() => onNodeClick?.(`block.${selectedBlockIdx}`, node)}
                  />

                  {/* Keep block selection behavior unchanged; only its visual
                      placement and sizing move into the selected stage. When
                      a child is selected, changing blocks must also remap the
                      Inspector selection to that block. See the 2026-07-14
                      entry in docs/DESIGN_DECISIONS.md. */}
                  <div className="arch-block-selector" aria-label="Choose transformer block">
                    {Array.from({ length: node.repeat_count }).map((_, idx) => (
                      <button
                        key={idx}
                        className={`arch-block-selector__button${selectedBlockIdx === idx ? " is-active" : ""}`}
                        aria-pressed={selectedBlockIdx === idx}
                        onClick={() => {
                          setSelectedBlockIdx(idx);
                          const m = selectedNodeId?.match(/^block\.\d+\.(.+)$/);
                          if (m && node.children) {
                            const child = node.children.find((c) => c.id === `block.{i}.${m[1]}`);
                            if (child) onNodeClick?.(`block.${idx}.${m[1]}`, child);
                          }
                        }}
                      >
                        {idx + 1}
                      </button>
                    ))}
                  </div>
                </div>
                <Arrow />
              </div>
            );
          }

          return (
            <div key={node.id} className="arch-flow-item">
              <div className="arch-stage">
                <NodeBox
                  label={node.id === "final_norm" ? "Final\nLayerNorm" : displayLabel(node.label)}
                  fullLabel={node.label}
                  kind={node.kind}
                  isSelected={selectedNodeId === node.id}
                  onClick={() => onNodeClick?.(node.id, node)}
                />
              </div>
              <Arrow />
            </div>
          );
        })}
        {nodes.length > 0 && (
          <div className="arch-flow-item arch-flow-item--io">
            <div className="arch-stage">
              <NodeBox label="Output" kind="io" />
            </div>
          </div>
        )}
      </div>
      </FitScale>

      {/* Second-level diagram for the selected block — per
          docs/Model_Diagram.md: "Inside a selected transformer block, show
          a second-level diagram." Real node IDs (block.{i}.ln1 etc.) so
          clicks actually match captured diagnostic data, not just the
          block-group placeholder id used above. MoE's expert layer is one
          clickable box (segmented visual, not 3 separate boxes) — it's
          captured diagnostically as a single opaque node, so 3 identical
          clickable boxes was misleading (implied per-expert data that
          doesn't exist). Direct user report, 2026-07-15. See
          docs/DESIGN_DECISIONS.md. */}
      {(() => {
        const blockGroup = nodes.find(isTransformerBlockGroup);
        if (!blockGroup?.children) return null;
        // Mirrors the fixed ln1/attention/ln2/mlp-or-moe index assumption
        // ResidualStreamOverlay already hardcodes (geometry.get(0..3)) — when
        // it doesn't hold (e.g. an RNN template's children), the overlay
        // renders nothing and the plain arrows below should stay untouched.
        const showResidualPath = blockGroup.children.length === 4;
        const headingScale = fitNaturalWidth ? fitScale : 1;
        return (
          <div className="arch-block-detail">
            <div
              className="arch-block-detail__heading-fit"
              style={{ height: 34 * headingScale }}
            >
              <div
                className="arch-block-detail__heading"
                style={{
                  width: fitNaturalWidth ?? "100%",
                  transform: `translateX(${fitInset}px) scale(${headingScale})`,
                }}
              >
                <span className="arch-block-detail__branch" aria-hidden="true">↳</span>
                <span>Inside selected transformer block</span>
                <strong>Block {selectedBlockIdx + 1}</strong>
              </div>
            </div>
            <div className="arch-block-detail__fit" style={{ height: fitHeight ?? undefined }}>
            <div
              className="arch-block-detail__flow"
              ref={detailFlowContainerRef}
              style={{ transform: `translateX(${fitInset}px) scale(${fitScale})` }}
            >
              {/* Residual-stream overlay. */}
              <ResidualStreamOverlay
                geometry={residualGeometry}
                entryArrow={entryArrowGeometry}
                exitArrow={exitArrowGeometry}
                containerHeight={100}
              />

              {/* Leading arrow — data entering the block. */}
              <Arrow variant="entry" elementRef={entryArrowRef} />

              {blockGroup.children.map((child, i) => {
                const nodeId = child.id.replace("{i}", String(selectedBlockIdx));
                const isMoe = child.kind === "moe";
                return (
                  <div
                    key={child.id}
                    className="arch-flow-item arch-flow-item--detail"
                  >
                    <div className="arch-stage">
                      <NodeBox
                        elementRef={(el) => {
                          if (el) {
                            detailBoxRefs.current.set(i, el);
                          } else {
                            detailBoxRefs.current.delete(i);
                          }
                        }}
                        segmented={isMoe}
                        label={isMoe ? "Experts" : displayLabel(child.label)}
                        fullLabel={child.label}
                        kind={child.kind}
                        isSelected={selectedNodeId === nodeId}
                        onClick={() => onNodeClick?.(nodeId, child)}
                      />
                      <span className="arch-stage-caption" aria-hidden="true">
                        {DETAIL_STAGE_CAPTIONS[i]}
                      </span>
                    </div>
                    {/* Keep the attention→ln2 arrow in flex layout so its gap
                        never collapses. The overlay paints the visible line
                        through the operator ring; `muted` only hides this
                        arrow's pixels, not its reserved width. */}
                    {i < blockGroup.children!.length - 1 && (
                      <Arrow muted={showResidualPath && i === 1} />
                    )}
                  </div>
                );
              })}

              {/* Trailing arrow — data leaving the block. Muted when the
                  overlay is active: it draws the connector through the
                  second operator ring in this exact spot instead. */}
              <Arrow variant="exit" elementRef={exitArrowRef} muted={showResidualPath} />
            </div>
            </div>
          </div>
        );
      })()}

      {/* Summary stats */}
      {manifest && (
        <div className="arch-summary">
          <div>Template <strong>{manifest.template}</strong></div>
          <div>Parameters <strong>{manifest.param_count.toLocaleString()}</strong> total · <strong>{manifest.trainable_param_count.toLocaleString()}</strong> trainable</div>
        </div>
      )}
    </div>
  );
}
