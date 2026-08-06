# Architecture Visual Variants — Design QA

## Comparison Target

- Product surface: expanded transformer architecture diagram in the dark desktop lab UI.
- State: Baseline Tiny Transformer, Block 1 expanded, no diagnostic node selected.
- Browser viewport: 1280 × 1100 CSS px, device scale factor 1.
- Generated visual directions:
  - Minimal/professional: `/home/nodozi/.codex/generated_images/019fd296-a6db-74e3-99db-24e602a13584/exec-4b91f6dc-7bc5-4aed-9d3e-c0196ce41ff8.png` (1920 × 819).
  - Technical/modern: `/home/nodozi/.codex/generated_images/019fd296-a6db-74e3-99db-24e602a13584/exec-86f593ad-4154-4263-808a-1a383eeb6be1.png` (1983 × 793).
  - Educational/friendly: `/home/nodozi/.codex/generated_images/019fd296-a6db-74e3-99db-24e602a13584/exec-77db1d74-7f4e-44ec-ad03-a0483f5b7e42.png` (1774 × 887).
- Refined browser renders:
  - `evidence/architecture-variants/minimal-professional.png` (1128 × 439).
  - `evidence/architecture-variants/technical-modern.png` (1128 × 441).
  - `evidence/architecture-variants/educational-friendly.png` (1128 × 487).
- Full-view comparison: `evidence/architecture-variants/all-variants-comparison.png`.
- Focused source-to-render comparison: `evidence/architecture-variants/source-vs-rendered-qa.png`.
- Density normalization: all browser renders are 1×. Source and implementation were compared at equal displayed widths; generated directions are intentionally taller concept frames, while the implementation preserves the requested compact architecture-panel footprint.

## Findings

- No actionable P0, P1, or P2 findings remain.
- [P3] The technical and educational generated directions include decorative node icons. The implementation uses restrained text codes only in the technical expanded view and no new iconography elsewhere. This keeps information content unchanged and avoids introducing a new icon dependency for a purely aesthetic exploration.
- [P3] The educational implementation is less saturated than its generated direction. This is intentional and directly supports the user's “fewer, calmer colours” priority.

## Required Fidelity Surfaces

- Fonts and typography: All variants retain the application's system sans serif for readable node labels. Technical micro-labels use the existing monospace token. The educational captions were increased to 11px after the first render; no essential text is tiny, clipped, or truncated.
- Spacing and layout rhythm: All final variants retain a 70px attention→LayerNorm gap at 1280px, with 28px addition operators centered in that space. No detail-panel overflow occurs. Nodes, residual junctions, and arrows share one centerline.
- Colors and visual tokens: Minimal is near-monochrome, technical uses restrained blue/mauve/cyan accents, and educational uses desaturated sand/mauve/teal categories. None uses gradients or glow.
- Image quality and asset fidelity: The diagram is code-native and contains no raster imagery, logos, illustrations, or placeholder assets. Browser captures are sharp at 1× density.
- Copy and content: Architecture labels, block badge, pipeline order, residual label, summary, and block selector remain unchanged. Educational stage captions restate existing node categories without adding model data.

## First-Render Critique And Refinement History

1. Minimal/professional first render: hierarchy and palette were calm, but the 22px addition operators were visually weak relative to 64px nodes. Refinement: increased both operators to 28px and raised residual-label size to 10px.
2. Technical/modern first render: detail-node type codes supported the instrument-panel direction, but the same codes cluttered compact top-pipeline nodes and caused avoidable wrapping. Refinement: confined type codes to the expanded block and increased operator size to 28px.
3. Educational/friendly first render: category colours and stage rhythm worked, but 10px captions were too small. Refinement: increased captions to 11px/650 weight and standardized 28px operators.
4. Post-refinement visual evidence: all three variants preserve clear gaps, align connectors through operator centers, fit without detail-panel overflow, and produce no browser console or page errors.

## Interaction And Runtime Checks

- Block selector remains keyboard/clickable product UI; focused unit coverage verifies selection remapping and unchanged labels.
- Node click behavior remains unchanged; the residual SVG retains `pointer-events: none`.
- Variant review uses URL parameters only and introduces no visible control or production workflow.
- Unknown variant values fall back to minimal/professional.
- Browser console and page errors checked across all three refined renders; none found.

## Implementation Checklist

- [x] Minimal/professional implemented, rendered, critiqued, and refined.
- [x] Technical/modern implemented, rendered, critiqued, and refined.
- [x] Educational/friendly implemented, rendered, critiqued, and refined.
- [x] Attention and LayerNorm remain visibly separated.
- [x] Residual rails, branch points, connectors, and addition operators are clear.
- [x] Selected and expanded states are restrained and variant-appropriate.
- [x] Information and behavior remain stable across variants.
- [x] Screenshots and comparison evidence persisted in the workspace.

final result: passed
