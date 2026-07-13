import { useState, useEffect } from "react";
import { ArchitectureManifest, ArchitectureNode } from "../types";
import { fetchArchitecture } from "../hooks/useApi";

interface Props {
  runId?: number | null;
  onNodeClick?: (nodeId: string, node: ArchitectureNode) => void;
  selectedNodeId?: string | null;
}

interface BoxProps {
  label: string;
  kind: string;
  isSelected?: boolean;
  onClick?: () => void;
  small?: boolean;
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

function NodeBox({ label, kind, isSelected, onClick, small }: BoxProps) {
  const colors = COLOR_MAP[kind] || { bg: "#e5e7eb", border: "#6b7280" };

  return (
    <div
      onClick={onClick}
      style={{
        minWidth: small ? 60 : 120,
        padding: small ? "6px 6px" : "12px 10px",
        borderRadius: 6,
        backgroundColor: colors.bg,
        border: `2px solid ${colors.border}`,
        cursor: onClick ? "pointer" : "default",
        borderColor: isSelected ? "#ff6b6b" : colors.border,
        boxShadow: isSelected ? "0 0 8px rgba(255, 107, 107, 0.5)" : "none",
        textAlign: "center",
        fontSize: small ? 10 : 12,
        fontWeight: 500,
        color: "#1a202c",
        transition: "all 0.15s",
      }}
    >
      {label}
    </div>
  );
}

function Arrow() {
  return (
    <div
      style={{
        width: 20,
        height: 2,
        backgroundColor: "#666",
        margin: "0 4px",
      }}
    />
  );
}

export default function ArchSchematic({ runId, onNodeClick, selectedNodeId }: Props) {
  const [manifest, setManifest] = useState<ArchitectureManifest | null>(null);
  const [selectedBlockIdx, setSelectedBlockIdx] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (runId) {
      setLoading(true);
      fetchArchitecture(runId)
        .then(setManifest)
        .catch(() => setManifest(null))
        .finally(() => setLoading(false));
    }
  }, [runId]);

  if (loading) {
    return <div className="panel"><h3>Architecture</h3><p>Loading...</p></div>;
  }

  const nodes = manifest ? manifest.nodes : [];

  return (
    <div className="panel" id="arch-schematic">
      <h3>Architecture</h3>

      {/* Horizontal pipeline diagram */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 0,
          overflowX: "auto",
          padding: "12px 0",
          marginBottom: 16,
        }}
      >
        {nodes.map((node, i) => {
          const isLast = i === nodes.length - 1;

          if (node.kind === "transformer_block_group" && node.children && node.repeat_count) {
            return (
              <div key={node.id} style={{ display: "flex", alignItems: "center" }}>
                <NodeBox
                  label={`Block ${selectedBlockIdx + 1} of ${node.repeat_count}`}
                  kind="transformer_block_group"
                  onClick={() => onNodeClick?.(`block.${selectedBlockIdx}`, node)}
                />

                {/* Block selector: small numbered buttons */}
                <div style={{ display: "flex", gap: 4, marginLeft: 8, marginRight: 8 }}>
                  {Array.from({ length: node.repeat_count }).map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedBlockIdx(idx)}
                      style={{
                        width: 24,
                        height: 24,
                        borderRadius: 4,
                        border: selectedBlockIdx === idx ? "2px solid #0284c7" : "1px solid #999",
                        background: selectedBlockIdx === idx ? "#0284c7" : "#f0f0f0",
                        color: selectedBlockIdx === idx ? "#fff" : "#333",
                        cursor: "pointer",
                        fontSize: 11,
                        fontWeight: 600,
                        transition: "all 0.15s",
                      }}
                    >
                      {idx + 1}
                    </button>
                  ))}
                </div>
                {!isLast && <Arrow />}
              </div>
            );
          }

          return (
            <div key={node.id} style={{ display: "flex", alignItems: "center" }}>
              <NodeBox
                label={node.label}
                kind={node.kind}
                isSelected={selectedNodeId === node.id}
                onClick={() => onNodeClick?.(node.id, node)}
              />
              {!isLast && <Arrow />}
            </div>
          );
        })}
      </div>

      {/* Second-level diagram for the selected block — per
          docs/Model_Diagram.md: "Inside a selected transformer block, show
          a second-level diagram." Real node IDs (block.{i}.ln1 etc.) so
          clicks actually match captured diagnostic data, not just the
          block-group placeholder id used above. MoE's expert layer renders
          as a few small boxes instead of one — same color as a dense MLP,
          "doesn't need to look very different" (user, 2026-07-13), just
          more of them. */}
      {(() => {
        const blockGroup = nodes.find((n) => n.kind === "transformer_block_group");
        if (!blockGroup?.children) return null;
        return (
          <div style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 12, marginBottom: 16 }}>
            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
              Inside Block {selectedBlockIdx + 1}:
            </span>
            {blockGroup.children.map((child, i) => {
              const nodeId = child.id.replace("{i}", String(selectedBlockIdx));
              const isMoe = child.kind === "moe";
              return (
                <div key={child.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  {isMoe ? (
                    <div style={{ display: "flex", gap: 2 }}>
                      {[0, 1, 2].map((expertIdx) => (
                        <NodeBox
                          key={expertIdx}
                          small
                          label={expertIdx === 1 ? "Experts" : ""}
                          kind="moe"
                          isSelected={selectedNodeId === nodeId}
                          onClick={() => onNodeClick?.(nodeId, child)}
                        />
                      ))}
                    </div>
                  ) : (
                    <NodeBox
                      small
                      label={child.label}
                      kind={child.kind}
                      isSelected={selectedNodeId === nodeId}
                      onClick={() => onNodeClick?.(nodeId, child)}
                    />
                  )}
                  {i < blockGroup.children!.length - 1 && <Arrow />}
                </div>
              );
            })}
          </div>
        );
      })()}

      {/* Summary stats */}
      {manifest && (
        <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
          <div>Template: <strong>{manifest.template}</strong></div>
          <div>Parameters: <strong>{manifest.param_count.toLocaleString()}</strong> total, <strong>{manifest.trainable_param_count.toLocaleString()}</strong> trainable</div>
        </div>
      )}
    </div>
  );
}
