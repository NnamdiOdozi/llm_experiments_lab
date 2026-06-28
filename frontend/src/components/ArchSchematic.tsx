import { ExperimentConfig } from "../types";

interface Props {
  config: ExperimentConfig;
}

function TransformerDiagram({ config }: { config: ExperimentConfig }) {
  const m = config.model;
  const posEnc = String(m.pos_encoding || "learned");
  const nLayers = Number(m.n_layer || 4);

  return (
    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.8 }}>
      <div style={{ color: "var(--text-dim)" }}>Input Tokens</div>
      <div>  |</div>
      <div style={{ color: "var(--accent)" }}>Token Embedding (vocab → {String(m.n_embd || 64)})</div>
      <div>  +</div>
      <div style={{ color: posEnc === "rope" ? "var(--yellow)" : "var(--green)" }}>
        Pos Encoding: {posEnc.toUpperCase()}
      </div>
      <div>  |</div>
      {Array.from({ length: nLayers }, (_, i) => (
        <div key={i}>
          <div style={{ color: "var(--accent)" }}>
            Block {i + 1}: MultiHead Attn ({String(m.n_head || 4)} heads) → FFN
          </div>
          <div>  |</div>
        </div>
      ))}
      <div style={{ color: "var(--green)" }}>LayerNorm → Linear → Logits</div>
    </div>
  );
}

function MoeDiagram({ config }: { config: ExperimentConfig }) {
  const m = config.model;
  const posEnc = String(m.pos_encoding || "rope");
  const nLayers = Number(m.n_layer || 4);
  const numExperts = Number(m.num_experts || 8);
  const topK = Number(m.top_k || 2);

  return (
    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.8 }}>
      <div style={{ color: "var(--text-dim)" }}>Input Tokens</div>
      <div>  |</div>
      <div style={{ color: "var(--accent)" }}>Token Embedding (vocab → {String(m.n_embd || 192)})</div>
      <div>  +</div>
      <div style={{ color: posEnc === "rope" ? "var(--yellow)" : "var(--green)" }}>
        Pos Encoding: {posEnc.toUpperCase()}
      </div>
      <div>  |</div>
      {Array.from({ length: nLayers }, (_, i) => (
        <div key={i}>
          <div style={{ color: "var(--accent)" }}>
            Block {i + 1}: MultiHead Attn ({String(m.n_head || 6)} heads)
          </div>
          <div style={{ color: "var(--yellow)", paddingLeft: 16 }}>
            → MoE Router → top-{topK} of {numExperts} experts
          </div>
          <div>  |</div>
        </div>
      ))}
      <div style={{ color: "var(--green)" }}>LayerNorm → Linear → Logits</div>
    </div>
  );
}

function RnnDiagram({ config }: { config: ExperimentConfig }) {
  const m = config.model;
  return (
    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.8 }}>
      <div style={{ color: "var(--text-dim)" }}>Input Characters (one-hot)</div>
      <div>  |</div>
      <div style={{ color: "var(--accent)" }}>
        LSTM ({String(m.n_layers || 2)} layers, hidden={String(m.n_hidden || 128)})
      </div>
      <div>  |</div>
      <div style={{ color: "var(--yellow)" }}>Dropout ({String(m.dropout || 0.3)})</div>
      <div>  |</div>
      <div style={{ color: "var(--green)" }}>Linear → Logits (vocab_size)</div>
    </div>
  );
}

export default function ArchSchematic({ config }: Props) {
  return (
    <div className="panel">
      <h3>Architecture</h3>
      {config.template === "rnn" ? (
        <RnnDiagram config={config} />
      ) : config.template === "moe" ? (
        <MoeDiagram config={config} />
      ) : (
        <TransformerDiagram config={config} />
      )}
    </div>
  );
}
