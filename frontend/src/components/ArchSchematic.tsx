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

function NodeBox({ label, fullLabel, kind, isSelected, isExpanded, onClick, segmented }: BoxProps) {
  const colors = COLOR_MAP[kind] || { bg: "#e5e7eb", border: "#6b7280" };

  return (
    <div
      className={`arch-node${isSelected ? " arch-node--selected" : ""}${isExpanded ? " arch-node--expanded" : ""}`}
      onClick={onClick}
      title={fullLabel && fullLabel !== label ? fullLabel : undefined}
      style={{
        backgroundColor: colors.bg,
        border: `2px solid ${colors.border}`,
        cursor: onClick ? "pointer" : "default",
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
      {label}
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

// Centralized drawing dimensions keep the residual diagram internally
// consistent and make future density changes a one-place edit.
const RESIDUAL_DIAGRAM = {
  streamY: 26,
  branchRadius: 5,
  operatorRadius: 11,
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
 * split into the attention sublayer and the bypass; the first addition then
 * becomes the source tapped by the MLP sublayer. Drawing one continuous upper
 * rail makes that sequence clearer than two unrelated brackets.
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
  const inputBranchX = entryArrow.left;
  const attentionAddX = (attention.right + ln2.left) / 2;
  const outputAddX = exitArrow.centerX;

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
        d={streamPath(attentionAddX, outputAddX)}
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

      {/* The first dot is the only pure divergence point. The top dot over the
          first addition shows where its result becomes the next bypass. */}
      <circle
        className="arch-residual-overlay__branch"
        cx={inputBranchX}
        cy={centerlineY}
        r={RESIDUAL_DIAGRAM.branchRadius}
        data-testid="residual-branch-dot"
      />
      <circle
        className="arch-residual-overlay__branch"
        cx={attentionAddX}
        cy={RESIDUAL_DIAGRAM.streamY}
        r={RESIDUAL_DIAGRAM.branchRadius}
        data-testid="residual-branch-dot"
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
      const containerRect = container.getBoundingClientRect();

      // Measure detail boxes.
      const geometry = new Map<number, BoxGeometry>();
      refs.forEach((el, idx) => {
        const rect = el.getBoundingClientRect();
        geometry.set(idx, {
          left: rect.left - containerRect.left,
          right: rect.right - containerRect.left,
          centerX: rect.left - containerRect.left + rect.width / 2,
          top: rect.top - containerRect.top,
          bottom: rect.bottom - containerRect.top,
          height: rect.height,
        });
      });
      setResidualGeometry(geometry);

      // Measure entry arrow position (left edge of flow container or arrow element).
      if (entryArrow) {
        const entryRect = entryArrow.getBoundingClientRect();
        setEntryArrowGeometry({
          left: entryRect.left - containerRect.left,
          right: entryRect.right - containerRect.left,
          centerX: entryRect.left - containerRect.left + entryRect.width / 2,
        });
      } else {
        setEntryArrowGeometry({
          left: 0,
          right: 0,
          centerX: 0,
        });
      }

      // Measure exit arrow position (right edge of flow container or arrow element).
      if (exitArrow) {
        const exitRect = exitArrow.getBoundingClientRect();
        setExitArrowGeometry({
          left: exitRect.left - containerRect.left,
          right: exitRect.right - containerRect.left,
          centerX: exitRect.left - containerRect.left + exitRect.width / 2,
        });
      } else {
        setExitArrowGeometry({
          left: containerRect.width,
          right: containerRect.width,
          centerX: containerRect.width,
        });
      }
    };

    // Measure immediately and on resize.
    measureGeometry();
    const handleResize = () => measureGeometry();
    window.addEventListener("resize", handleResize);

    // ResizeObserver is not available in jsdom; gracefully skip if missing.
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => measureGeometry());
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
    <div className="panel" id="arch-schematic">
      <h3>Architecture</h3>

      {/* Horizontal pipeline diagram */}
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
        return (
          <div className="arch-block-detail">
            <div className="arch-block-detail__heading">
              <span className="arch-block-detail__branch" aria-hidden="true">↳</span>
              <span>Inside selected transformer block</span>
              <strong>Block {selectedBlockIdx + 1}</strong>
            </div>
            <div className="arch-block-detail__flow" ref={detailFlowContainerRef}>
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
                    <div
                      className="arch-stage"
                      ref={(el) => {
                        if (el) {
                          detailBoxRefs.current.set(i, el);
                      } else {
                          detailBoxRefs.current.delete(i);
                        }
                      }}
                    >
                      <NodeBox
                        segmented={isMoe}
                        label={isMoe ? "Experts" : displayLabel(child.label)}
                        fullLabel={child.label}
                        kind={child.kind}
                        isSelected={selectedNodeId === nodeId}
                        onClick={() => onNodeClick?.(nodeId, child)}
                      />
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
