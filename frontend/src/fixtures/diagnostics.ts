// Demo-mode fixture data (?use_fixtures=true), moved out of useApi.ts.
// IMPORTANT: only load this file via dynamic import — `await
// import("../fixtures/diagnostics")` — inside a useFixtures() branch.
// A static top-level import would pull ~230 lines of demo data back into
// the main bundle for every real user; the dynamic import keeps it in a
// separate chunk that only demo sessions ever download. See
// docs/DESIGN_CHOICES / Fable Codebase Review "Code bloat / DRY".

// Fixture response for POST /diagnostics/start
export const FIXTURE_START_RESPONSE = {
  diagnostic_session_id: "diag-17",
  tokens: [
    { position: 0, id: 51, text: "The" },
    { position: 1, id: 82, text: " king" },
    { position: 2, id: 44, text: " said" }
  ]
};

// Fixture data for development/testing without a real backend
export const FIXTURE_MANIFEST: import("../types").ArchitectureManifest = {
  schema_version: 1,
  local_run_id: 42,
  template: "transformer",
  param_count: 1782529,
  trainable_param_count: 1782529,
  nodes: [
    {
      id: "embedding",
      kind: "embedding",
      label: "Token + Positional Embedding",
      config: { vocab_size: 65, n_embd: 192, pos_encoding: "learned" },
      static_shapes: [
        { name: "input", dims: ["batch", "sequence"] },
        { name: "output", dims: ["batch", "sequence", "n_embd"] }
      ],
      math_key: "embedding_lookup"
    },
    {
      id: "block",
      kind: "transformer_block_group",
      label: "Transformer Block",
      repeat_count: 4,
      config: { n_head: 6, head_dim: 32, dropout: 0.1, activation: "gelu" },
      children: [
        { id: "block.{i}.ln1", kind: "layernorm", label: "LayerNorm (pre-attention)", config: {} },
        { id: "block.{i}.attention", kind: "attention", label: "Causal Self-Attention", config: {},
          math_key: "scaled_dot_product_attention" },
        { id: "block.{i}.ln2", kind: "layernorm", label: "LayerNorm (pre-MLP)", config: {} },
        { id: "block.{i}.mlp", kind: "mlp", label: "Feed-Forward (dense)", config: {},
          math_key: "mlp_gelu" }
      ]
    },
    {
      id: "final_norm",
      kind: "layernorm",
      label: "Final LayerNorm",
      config: {}
    },
    {
      id: "lm_head",
      kind: "lm_head",
      label: "LM Head",
      config: { vocab_size: 65 },
      static_shapes: [
        { name: "input", dims: ["batch", "sequence", "n_embd"] },
        { name: "output", dims: ["batch", "sequence", "vocab_size"] }
      ]
    }
  ]
};

// Fixture snapshot with attention data populated (Phase 2)
export const FIXTURE_SNAPSHOT_WITH_ATTENTION: import("../types").DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-17",
  generation_step: 3,
  input_tokens: [
    { position: 0, id: 51, text: "The" },
    { position: 1, id: 82, text: " king" },
    { position: 2, id: 44, text: " said" }
  ],
  generated_token: { position: 3, id: 91, text: " to" },
  nodes: {
    "embedding": {
      input_shape: [1, 3],
      output_shape: [1, 3, 192],
      summary: { mean: 0.004, std: 0.131, l2_norm: 3.27, min: -0.51, max: 0.62 }
    },
    "block.0.ln1": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.1, max: 3.4 }
    },
    "block.0.attention": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.01, std: 0.44, l2_norm: 10.8, min: -1.9, max: 2.1 }
    },
    "block.0.ln2": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.0, max: 3.2 }
    },
    "block.0.mlp": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.02, std: 0.51, l2_norm: 12.3, min: -2.2, max: 2.4 }
    },
    "final_norm": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.2, max: 3.3 }
    }
  },
  attention: {
    available: true,
    layer: 0,
    head: 0,
    weights: [
      [1.0, 0.0, 0.0],
      [0.62, 0.38, 0.0],
      [0.21, 0.35, 0.44]
    ],
    token_labels: ["The", " king", " said"],
    qkv_detail: {
      positions: [0, 1, 2],
      tokens: ["The", " king", " said"],
      q: [
        [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03],
        [0.14, -0.30, 0.10, 0.48, -0.11, 0.20, -0.40, 0.05],
        [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03],
      ],
      k: [
        [0.08, 0.21, -0.17, 0.24, 0.10, -0.38, 0.13, 0.19],
        [0.09, 0.19, -0.15, 0.22, 0.12, -0.35, 0.14, 0.17],
        [0.08, 0.21, -0.17, 0.24, 0.10, -0.38, 0.13, 0.19],
      ],
      v: [
        [-0.15, 0.42, 0.05, -0.29, 0.18, 0.31, -0.22, 0.13],
        [-0.13, 0.39, 0.07, -0.27, 0.16, 0.29, -0.20, 0.11],
        [-0.15, 0.42, 0.05, -0.29, 0.18, 0.31, -0.22, 0.13],
      ],
    }
  },
  activation_summaries: {
    available: true,
    top_abs_components: [
      { index: 42, value: 3.1 },
      { index: 17, value: -2.8 },
      { index: 105, value: 2.4 }
    ],
    value_slice: [0.12, -0.34, 0.08, 0.51, -0.09, 0.22, -0.44, 0.03]
  },
  lm_head: {
    logits_shape: [1, 3, 65],
    selected_position: 2,
    top_k: [
      { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
      { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
      { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
      { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
    ],
    top_k_by_position: [
      { position: 0, token: "The", actual_next_token_id: 82, top_k: [
        { rank: 1, token_id: 82, token: " king", logit: 5.10, probability: 0.28 },
        { rank: 2, token_id: 44, token: " said", logit: 4.80, probability: 0.19 },
      ] },
      { position: 1, token: " king", actual_next_token_id: 44, top_k: [
        { rank: 1, token_id: 44, token: " said", logit: 5.50, probability: 0.33 },
        { rank: 2, token_id: 91, token: " to", logit: 4.90, probability: 0.20 },
      ] },
      { position: 2, token: " said", actual_next_token_id: 91, top_k: [
        { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
        { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
        { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
        { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
        { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
      ] },
    ]
  },
  position_tokens: [],
  complete: true
};

// Fixture snapshot without attention (Phase 1 style, for backward compat)
export const FIXTURE_SNAPSHOT: import("../types").DiagnosticSnapshot = {
  schema_version: 1,
  diagnostic_session_id: "diag-17",
  generation_step: 3,
  input_tokens: [
    { position: 0, id: 51, text: "The" },
    { position: 1, id: 82, text: " king" },
    { position: 2, id: 44, text: " said" }
  ],
  generated_token: { position: 3, id: 91, text: " to" },
  nodes: {
    "embedding": {
      input_shape: [1, 3],
      output_shape: [1, 3, 192],
      summary: { mean: 0.004, std: 0.131, l2_norm: 3.27, min: -0.51, max: 0.62 }
    },
    "block.0.ln1": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.1, max: 3.4 }
    },
    "block.0.attention": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.01, std: 0.44, l2_norm: 10.8, min: -1.9, max: 2.1 }
    },
    "block.0.ln2": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.0, max: 3.2 }
    },
    "block.0.mlp": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.02, std: 0.51, l2_norm: 12.3, min: -2.2, max: 2.4 }
    },
    "final_norm": {
      input_shape: [1, 3, 192],
      output_shape: [1, 3, 192],
      summary: { mean: 0.0, std: 1.0, l2_norm: 24.1, min: -3.2, max: 3.3 }
    }
  },
  attention: { available: false, reason: "Not requested" },
  activation_summaries: { available: false, reason: "Not requested" },
  lm_head: {
    logits_shape: [1, 3, 65],
    selected_position: 2,
    top_k: [
      { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
      { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      { rank: 3, token_id: 33, token: " be", logit: 5.40, probability: 0.14 },
      { rank: 4, token_id: 7,  token: " see", logit: 5.02, probability: 0.09 },
      { rank: 5, token_id: 58, token: " go",  logit: 4.75, probability: 0.07 }
    ],
    top_k_by_position: [
      { position: 2, token: " said", actual_next_token_id: 91, top_k: [
        { rank: 1, token_id: 91, token: " to", logit: 6.21, probability: 0.31 },
        { rank: 2, token_id: 12, token: " have", logit: 5.87, probability: 0.22 },
      ] },
    ]
  },
  position_tokens: [],
  complete: true
};
