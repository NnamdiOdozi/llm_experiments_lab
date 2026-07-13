ent.

Briefing Note 1: React Model Visualisation and Interactive Diagnostics UI
Objective

Build a horizontal, clickable model visualisation beneath the loss curves. It should help a beginner answer four questions:

What components does this model contain?
What happens to the tensor shape as data passes through them?
What mathematical operation does each component perform?
When the model is paused and prompted, what happened during this particular forward pass?

The visualisation must work in two distinct modes:

Training mode
    Model structure, configuration and high-level progress

Paused diagnostic mode
    Runtime tensor shapes, attention, activations and top-k outputs

Do not attempt to stream every activation during training. That would create a noisy interface, large payloads and unnecessary trainer overhead.

Primary layout decision

The project discussion proposed a vertical “stackable layer list.” That is natural conceptually, but the current page does not have enough vertical room below the loss curves. The main visualisation should therefore be a horizontal computational pipeline.

Suggested structure:

Prompt/Tokens
    →
Token Embedding
    →
Positional Encoding
    →
Transformer Block × N
    →
Final LayerNorm
    →
LM Head
    →
Top-k Tokens

The visual should occupy the full available width beneath the chart. It may use horizontal scrolling on narrower screens rather than squeezing every box until the labels become unreadable.

Do not render all transformer blocks separately by default. A 12-layer model should not create 12 nearly identical boxes. Use a grouped node:

Transformer Block × 4

Clicking it opens a detailed view where the user can choose Block 1, Block 2, and so on.

Inside a selected transformer block, show a second-level diagram:

Input
  ├→ LayerNorm → Q/K/V → Attention → Projection ─┐
  └───────────────────────────────────────────────+→ Residual
                                                   ↓
             LayerNorm → MLP or MoE → Residual → Output

This is more accurate than implying the entire transformer is a simple nn.Sequential.

Important correction to the stackable-layer proposal

Do not initially allow arbitrary insertion, deletion and reordering of layers such as BatchNorm, Dropout and Linear anywhere in the transformer.

That sounds flexible but creates many invalid or misleading architectures:

dimensions may no longer match;
BatchNorm is not a normal transformer component;
moving LayerNorm changes whether the network is pre-norm or post-norm;
deleting residual-sensitive components may make the model unusable;
existing checkpoints will no longer load.

For the MVP, use constrained component choices:

Number of transformer blocks
Attention type
Positional encoding type
Dense MLP versus MoE
Embedding dimension
Number of heads
Dropout
Activation function
Vocabulary/tokenizer

The user should be able to select valid alternatives, not construct arbitrary graphs.

The architecture visualisation should reflect the config, but it should not become a free-form neural-network editor.

Structural settings versus runtime settings

Once training starts, lock settings that change tensor shapes or parameter structure:

embedding dimension
number of heads
number of layers
vocabulary size
attention implementation
dense versus MoE structure
expert count

Changing these would invalidate or partially invalidate the existing checkpoint.

The UI may allow the user to inspect them while paused, but not mutate the current run. An attempted structural change should offer:

Create a new experiment from this configuration

Some training settings can potentially change when paused:

learning rate
optimizer schedule
weight decay
dropout probability

Even those changes should be recorded as a run event because they change the experiment. Do not silently mutate them.

Generation settings can safely be adjusted in paused diagnostic mode:

temperature
top-k
top-p
maximum new tokens
sampling versus greedy
Click behaviour and detail panel

Each node should be clickable.

Clicking a node opens a right-hand details drawer. This drawer should share the same right-pane area used by the Lab Assistant rather than permanently consuming more page width.

Suggested right-pane tabs:

Assistant
Inspector
Events

Within Inspector:

Overview
Shapes
Math
Configuration
Runtime
Overview

Plain-language explanation:

Token Embedding replaces each token ID with a learned vector.

Shapes

Show named dimensions rather than only numbers:

Input token IDs:       [batch, sequence]
                       [1, 5]

Embedding output:      [batch, sequence, embedding]
                       [1, 5, 192]

Where practical, visually annotate which dimension is which.

Math

Show the mathematical operation and its dimensions.

Examples:

Embedding:
X = E[token_ids]

Q = XW_Q
K = XW_K
V = XW_V

Attention(Q,K,V)
= softmax(QKᵀ / √d_head)V

Do not dump formal notation without explanation. Each formula should have one plain-English sentence beneath it.

Configuration

Show parameters relevant to that node:

embedding_dim: 192
n_heads: 6
head_dim: 32
dropout: 0.1
attention_type: multi-head

Also show a diff from the selected baseline when available:

n_heads: 4 → 6
dropout: unchanged
Runtime

Only populated during paused diagnostic inspection. It may show:

actual input/output shapes
summary statistics
selected token position
attention heatmap
top-k values
Training-mode behaviour

During training, the visualisation is mainly structural.

It should show:

current architecture;
parameter count;
trainable parameter count;
active parameter count for MoE;
optimizer;
learning rate;
batch size;
block size;
device;
current run status.

A subtle state marker may show:

training
pause requested
paused
prompting
resuming
completed

Do not animate tokens flowing through layers during training unless the animation reflects real captured events. A decorative animation could falsely suggest that the UI is displaying the current batch.

Do not show raw weights. Raw weight arrays offer almost no educational value. A later advanced panel may show:

weight tensor shape
parameter count
mean/std
norm
histogram

but not thousands of raw values.

Paused diagnostic mode

The default diagnostic subject should be a fresh, short prompt, not the final training batch.

Example:

Hello

or:

The king said

This creates a clean explanatory path:

text
→ token IDs
→ embeddings
→ transformer states
→ logits
→ next token

Inspecting the last training batch may be added later as an advanced feature, but it is not suitable as the default. Batches contain multiple sequences and many tokens and will overwhelm a beginner.

Limit diagnostic prompts initially to perhaps 32 or 64 tokens.

Step-through generation

Support two controls:

>    Generate and inspect one next token

>>   Continue generation until max_new_tokens or stop condition

The > button should mean one autoregressive generation step, not one neural-network layer.

One button press runs a complete forward pass, selects the next token and records a diagnostic snapshot. The learner can then inspect each layer from that captured pass without rerunning each layer separately.

After several > presses, >> should continue from the existing generated sequence rather than restarting the prompt.

Display:

Prompt tokens
Generated tokens
Current selected token
Current generation step
Diagnostic views

Prioritise these in order:

Tensor shapes.
Tokenisation and embeddings.
Top-k LM-head results.
Attention heatmap.
Activation summaries.
Q/K/V details.
Gradients and weight summaries later.
Top-k LM head

For the selected next-token position, display:

rank
token
token ID
logit
probability

Default to top 5. Permit 5, 10 or 20, with a hard cap.

Attention

Show one layer and one head at a time initially. Provide selectors:

Layer: 2
Head: 3

Render a token-by-token heatmap with token labels on each axis.

Do not initially display every head from every layer simultaneously.

Vectors and activations

Raw 192- or 768-dimensional vectors are usually not useful as tables.

Show:

shape
L2 norm
mean
standard deviation
minimum
maximum
largest absolute components
small heat strip of values

Allow an explicit “show raw values” expansion only for short vectors.

Frontend data model

The frontend should consume a backend-provided architecture manifest rather than reverse-engineering the Python model from component names.

Suggested shape:

interface ArchitectureNode {
  id: string;
  kind: string;
  label: string;
  repeatCount?: number;
  enabled: boolean;
  config: Record<string, unknown>;
  staticShapes?: ShapeDescription[];
  mathKey?: string;
  children?: ArchitectureNode[];
}

Runtime data should attach to nodes by stable node_id.

Do not couple the visualisation directly to one hard-coded Tiny Transformer class. The same visual components should later support RoPE, MoE and perhaps RNNs.

Acceptance criteria

The visualisation is not complete until:

it renders a horizontal overview under the loss curves;
repeated transformer blocks are grouped;
nodes open a details panel;
the details panel shows overview, shapes, math and config;
paused prompt inspection populates runtime information;
one-step and continue-generation controls are clearly distinguished;
structural settings are locked during an existing run;
the UI does not attempt to display full raw weights;
the layout remains usable at desktop and laptop widths;
missing diagnostic data produces “Not captured” rather than an error.


Addendum: model diagram and additional trainer metrics

Use a single expanded decoder block as the main architecture visual, inspired by the attached Bycroft-style diagram:

Tokens → Embedding + Position
       → Transformer Block × N
       → Final LayerNorm → LM Head → Softmax/Top-k

Show one representative transformer block containing:

LayerNorm → Causal Self-Attention → Residual
LayerNorm → MLP or MoE           → Residual

Do not draw all blocks. Label the block × 4, × 8, etc. Add a selector such as Block 1 of 4 so paused diagnostics can display data captured from a chosen block. Highlight the currently selected component and open its Shapes/Math/Config/Runtime details in the Inspector pane.