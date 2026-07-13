import { describe, it, expect } from 'vitest';
import {
  ArchitectureManifest,
  ArchitectureNode,
  DiagnosticSnapshot,
  TopKEntry,
  NodeShape,
} from '../types';

describe('Diagnostic Types', () => {
  it('should create a valid ArchitectureManifest', () => {
    const manifest: ArchitectureManifest = {
      schema_version: 1,
      local_run_id: 42,
      template: 'transformer',
      param_count: 1782529,
      trainable_param_count: 1782529,
      nodes: [
        {
          id: 'embedding',
          kind: 'embedding',
          label: 'Token Embedding',
          config: { vocab_size: 65 },
        },
        {
          id: 'lm_head',
          kind: 'lm_head',
          label: 'LM Head',
          config: { vocab_size: 65 },
        },
      ],
    };

    expect(manifest.schema_version).toBe(1);
    expect(manifest.nodes.length).toBe(2);
    expect(manifest.nodes[0].kind).toBe('embedding');
  });

  it('should support block group with children', () => {
    const blockNode: ArchitectureNode = {
      id: 'block',
      kind: 'transformer_block_group',
      label: 'Transformer Block',
      repeat_count: 4,
      config: { n_head: 6 },
      children: [
        {
          id: 'block.{i}.ln1',
          kind: 'layernorm',
          label: 'LayerNorm',
          config: {},
        },
        {
          id: 'block.{i}.attention',
          kind: 'attention',
          label: 'Attention',
          config: {},
        },
        {
          id: 'block.{i}.mlp',
          kind: 'mlp',
          label: 'MLP',
          config: {},
        },
      ],
    };

    expect(blockNode.repeat_count).toBe(4);
    expect(blockNode.children).toHaveLength(3);
  });

  it('should support MoE nodes', () => {
    const moeNode: ArchitectureNode = {
      id: 'block.0.moe',
      kind: 'moe',
      label: 'Mixture of Experts',
      config: { num_experts: 8, top_k: 2, capacity_factor: 1.25 },
    };

    expect(moeNode.kind).toBe('moe');
    expect(moeNode.config.num_experts).toBe(8);
  });

  it('should create a valid DiagnosticSnapshot', () => {
    const topK: TopKEntry[] = [
      { rank: 1, token_id: 91, token: ' to', logit: 6.21, probability: 0.31 },
      { rank: 2, token_id: 12, token: ' have', logit: 5.87, probability: 0.22 },
    ];

    const snapshot: DiagnosticSnapshot = {
      schema_version: 1,
      diagnostic_session_id: 'diag-17',
      generation_step: 3,
      input_tokens: [
        { position: 0, id: 51, text: 'The' },
        { position: 1, id: 82, text: ' king' },
      ],
      generated_token: { position: 2, id: 91, text: ' to' },
      nodes: {
        embedding: {
          input_shape: [1, 2],
          output_shape: [1, 2, 192],
          summary: { mean: 0.004, std: 0.131, l2_norm: 3.27, min: -0.51, max: 0.62 },
        },
      },
      attention: { available: false, reason: 'Deferred to phase 2' },
      activation_summaries: { available: false, reason: 'Deferred to phase 2' },
      lm_head: {
        logits_shape: [1, 2, 65],
        selected_position: 1,
        top_k: topK,
        top_k_by_position: [{ position: 1, token: ' king', top_k: topK }],
      },
      complete: true,
    };

    expect(snapshot.generation_step).toBe(3);
    expect(snapshot.lm_head.top_k).toHaveLength(2);
    expect(snapshot.lm_head.top_k[0].rank).toBe(1);
  });

  it('should support NodeShape with named dims', () => {
    const shape: NodeShape = {
      name: 'output',
      dims: ['batch', 'sequence', 'n_embd'],
    };

    expect(shape.dims).toContain('batch');
  });
});
