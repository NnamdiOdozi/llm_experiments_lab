# Architecture Restoration — Design QA

## Comparison target

- Source visual truth: `evidence/architecture-variants/minimal-professional.png` (1128 × 439 px), supplemented by the user's requirement that the residual stream now be two disjoint spans.
- Browser implementation: `evidence/architecture-variants/restored-responsive-1512.png` (700 × 327 px).
- Constrained-width evidence: `evidence/architecture-variants/restored-responsive-900.png` (418 × 200 px after the dashboard's 0.595238 global zoom).
- MoE evidence: `evidence/architecture-variants/restored-responsive-moe-1512.png` (700 × 327 px).
- Combined comparison input: `evidence/architecture-variants/restoration-source-comparison.png` (1600 × 900 px).
- State: dark minimal/professional architecture panel, Block 1 expanded, no diagnostic node selected.
- Browser viewport and density: 1512 × 1200 CSS px and 900 × 1200 CSS px, `deviceScaleFactor: 1`.
- Density normalization: source and implementation were shown at equal comparison-column widths in the combined evidence. The implementation's narrower native panel is intentional because it occupies the responsive centre dashboard column.

## Findings

- No actionable P0, P1, or P2 findings remain.
- [P3] Labels become small in the 900px side-pane simulation because the user's dashboard-wide uniform scaling applies after the architecture's explicit 90% safety fit. All labels remain present and unclipped.

## Required fidelity surfaces

- Fonts and typography: The system sans-serif, weights, wrapping, uppercase Architecture label, block heading, badge, and summary hierarchy match the established minimal variant. Text is unclipped at both tested widths.
- Spacing and layout rhythm: Input → Output remains visible as one centered pipeline. The expanded block retains both boundary arrows, four separated stages, two addition points, and two visibly disjoint residual spans. No inner horizontal scrollbar appears.
- Colors and visual tokens: Existing dark surfaces, muted blue-gray borders, restrained LayerNorm/attention/feed-forward categorization, and residual color tokens are unchanged.
- Image quality and asset fidelity: The diagram remains code-native and sharp at 1× density. There are no raster assets, generated illustrations, logos, or placeholder imagery in the component.
- Copy and content: Input, T + P Embedding, Transformer Block, Final LayerNorm, LM Head, Output, LayerNorm, Causal S. Attention, Feed Forward/Experts, block controls, and model summary remain intact.

## Comparison history

1. Initial constrained render: the rows fitted the available width, but residual geometry used each node's nearest positioned wrapper as its coordinate origin. This collapsed the first addition and first bypass near the block input (P1).
2. Fix: node and arrow rectangles are now normalized into the expanded flow's natural coordinate system, compensating for both the local fit transform and dashboard zoom. The panel's own horizontal overflow was removed.
3. First post-fix review: geometry and scrolling were correct, but the rightmost Output and Feed Forward/Experts nodes still sat too close to the visible boundary in the user's browser configuration (P1).
4. Refinement: both canvases now occupy at most 90% of their available width and are centered, producing explicit 5% safety insets instead of relying on edge-to-edge fitting.
5. Proportion refinement: the expanded-block heading and Block badge now use the same measured transform and horizontal frame as the inner diagram, eliminating the oversized chrome seen above the reduced flow.
6. Final evidence: the 1512px and 900px Transformer plus 1512px MoE captures show full Output and Feed Forward/Experts nodes, proportionate heading chrome, correct additions, two disjoint bypass spans, and no scrolling or clipping.

## Interaction and runtime checks

- Block 2 selection was clicked and confirmed in the heading, then returned to Block 1 before capture, at all three browser verification states.
- Browser console errors: none.
- Page errors: none.
- Architecture component tests: 14/14 passed.
- Full frontend suite: 102/102 passed.
- Production build: passed; only the existing Vite chunk-size advisory remains.

## Implementation checklist

- [x] Preserve the dashboard's responsive/global scaling behavior.
- [x] Fit the outer pipeline without clipping Input or Output.
- [x] Fit the expanded Transformer Block without an inner scrollbar.
- [x] Restore residual overlay geometry in transformed coordinates.
- [x] Preserve two disjoint residual spans and both addition points.
- [x] Verify both Feed Forward and Experts variants.
- [x] Verify full and constrained dashboard widths in Chromium.

final result: passed
