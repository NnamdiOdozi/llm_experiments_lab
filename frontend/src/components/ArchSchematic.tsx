import { useState, useEffect } from "react";
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

function Arrow() {
  return <div className="arch-arrow" aria-hidden="true" />;
}

export default function ArchSchematic({ runId, config, onNodeClick, selectedNodeId }: Props) {
  const [manifest, setManifest] = useState<ArchitectureManifest | null>(null);
  const [selectedBlockIdx, setSelectedBlockIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  // Tracks whether at least one fetch attempt has failed — distinguishes
  // "still loading" from "worker's cold-starting, retrying" in the UI.
  const [retrying, setRetrying] = useState(false);

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
        return (
          <div className="arch-block-detail">
            <div className="arch-block-detail__heading">
              <span className="arch-block-detail__branch" aria-hidden="true">↳</span>
              <span>Inside selected transformer block</span>
              <strong>Block {selectedBlockIdx + 1}</strong>
            </div>
            <div className="arch-block-detail__flow">
              {blockGroup.children.map((child, i) => {
                const nodeId = child.id.replace("{i}", String(selectedBlockIdx));
                const isMoe = child.kind === "moe";
                return (
                  <div key={child.id} className="arch-flow-item arch-flow-item--detail">
                    <div className="arch-stage">
                      <NodeBox
                        segmented={isMoe}
                        label={isMoe ? "Experts" : displayLabel(child.label)}
                        fullLabel={child.label}
                        kind={child.kind}
                        isSelected={selectedNodeId === nodeId}
                        onClick={() => onNodeClick?.(nodeId, child)}
                      />
                    </div>
                    {i < blockGroup.children!.length - 1 && <Arrow />}
                  </div>
                );
              })}
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
