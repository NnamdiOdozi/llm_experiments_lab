"""Export experiment as downloadable .py script or .ipynb notebook.

Assembles pre-written template code with user's config values baked in.
"""

import json
import nbformat
from pathlib import Path

from backend.training.templates.transformer.model import (
    MultiHeadSelfAttention,
    RotaryPositionalEncoding,
    FeedForward,
    Block,
    TinyTransformerLM,
)
from backend.training.templates.rnn.model import CharRNN
from config.settings import settings


def _get_tokenizer_type(config: dict) -> str:
    """Extract tokenizer type from config. Defaults to 'char'."""
    data_config = config.get("data", {})
    return data_config.get("tokenizer", "char")


def _get_tokenizer_artifact_path(config: dict) -> str | None:
    """Extract tokenizer artifact filename from config."""
    data_config = config.get("data", {})
    return data_config.get("tokenizer_artifact")


def _build_char_tokenizer_code(settings_obj) -> str:
    """Build char-based tokenizer setup code (original behavior)."""
    return f'''url = "{settings_obj.shakespeare_url}"
text = requests.get(url, timeout={settings_obj.http_timeout}).text
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {{ch: i for i, ch in enumerate(chars)}}
itos = {{i: ch for ch, i in stoi.items()}}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)
'''


def _build_bpe_tokenizer_code(config: dict, artifact_path: str) -> str:
    """Build BPE tokenizer setup code using huggingface tokenizers."""
    data_config = config.get("data", {})
    vocab_size = data_config.get("vocab_size", 1024)

    return f'''# Load BPE tokenizer from artifact
# NOTE: This export requires the tokenizer artifact file '{artifact_path}'
# alongside this script. It is included in the .zip export bundle.
from tokenizers import Tokenizer

tok = Tokenizer.from_file("{artifact_path}")
vocab_size = {vocab_size}

encode = lambda s: tok.encode(s).ids
decode = lambda ids: tok.decode(ids)

# Reconstruct text for train/val split (using original text)
url = "{settings.shakespeare_url}"
text = requests.get(url, timeout={settings.http_timeout}).text
'''


def _transformer_script(config: dict) -> str:
    """Assemble standalone transformer training script with config baked in."""
    m = config["model"]
    t = config["training"]
    pos = m.get("pos_encoding", "learned")
    use_rope = pos == "rope"

    tokenizer_type = _get_tokenizer_type(config)
    artifact_path = _get_tokenizer_artifact_path(config)

    script = f'''"""Tiny Transformer LM — exported from LLM Experiments Lab.

Config: {config.get("name", "custom experiment")}
Template: transformer | Pos encoding: {pos} | Tokenizer: {tokenizer_type}
"""

import math
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
'''

    # Add tokenizers import for BPE
    if tokenizer_type != "char":
        script += "\nfrom tokenizers import Tokenizer\n"

    script += f'''
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed({settings.random_seed})

# ── Hyperparameters (from your experiment config) ──
BLOCK_SIZE = {m["block_size"]}
N_EMBD = {m["n_embd"]}
N_HEAD = {m["n_head"]}
N_LAYER = {m["n_layer"]}
DROPOUT = {m["dropout"]}
POS_ENCODING = "{pos}"
BATCH_SIZE = {t["batch_size"]}
LEARNING_RATE = {t["learning_rate"]}
MAX_ITERS = {t["max_iters"]}
EVAL_INTERVAL = {t["eval_interval"]}
EVAL_ITERS = {t.get("eval_iters", 200)}
GENERATE_TOKENS = 500
TRAIN_VAL_SPLIT = 0.9


# ── Data ──
'''

    # Branch on tokenizer type
    if tokenizer_type == "char":
        script += _build_char_tokenizer_code(settings)
    else:
        script += _build_bpe_tokenizer_code(config, artifact_path or "tokenizer.json")

    script += f'''
data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(TRAIN_VAL_SPLIT * len(data))
train_data, val_data = data[:n_train], data[n_train:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

'''

    if use_rope:
        script += '''
# ── RoPE ──
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(block_size).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        T = x.shape[2]
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

'''

    script += f'''
# ── Attention ──
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
{"        self.rope = RotaryPositionalEncoding(self.head_size, block_size)" if use_rope else ""}

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
{"        q = self.rope(q)\\n        k = self.rope(k)" if use_rope else ""}
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = self.attn_dropout(torch.softmax(scores, dim=-1))
        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


# ── Feed-forward ──
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ──
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Full Model ──
class TinyTransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
{"        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)" if not use_rope else ""}
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
{"        x = tok + self.pos_emb(torch.arange(T, device=device))" if not use_rope else "        x = tok  # RoPE applied inside attention"}
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.train(False)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ── Training ──
model = TinyTransformerLM().to(device)
print(f"Parameters: {{sum(p.numel() for p in model.parameters()) / 1e6:.2f}}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    model.train(False)
    out = {{}}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(True)
    return out


for it in range(MAX_ITERS + 1):
    if it % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"iter {{it:5d}} | train {{losses['train']:.4f}} | val {{losses['val']:.4f}}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ── Generate ──
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\\n--- Generated text ---")
print(decode(model.generate(ctx, GENERATE_TOKENS)[0].tolist()))
'''

    return script


def _rnn_script(config: dict) -> str:
    """Assemble standalone RNN training script with config baked in."""
    m = config["model"]
    t = config["training"]

    return f'''"""CharRNN Language Model — exported from LLM Experiments Lab.

Config: {config.get("name", "custom experiment")}
Template: rnn (LSTM)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed({settings.random_seed})

# ── Hyperparameters (from your experiment config) ──
N_HIDDEN = {m.get("n_hidden", 256)}
N_LAYERS = {m.get("n_layers", 2)}
DROPOUT = {m.get("dropout", 0.5)}
BATCH_SIZE = {t["batch_size"]}
LEARNING_RATE = {t["learning_rate"]}
EPOCHS = {t.get("epochs", 50)}
SEQ_LEN = {t.get("seq_len", 50)}
CLIP = {t.get("clip", 5)}
TRAIN_VAL_SPLIT = 0.8
MAX_NEW_TOKENS = 200
NUM_SAMPLES = 10


# ── Data ──
# Download dinos.txt from: https://drive.google.com/uc?id=1SzD-jBAyLuakrbR4nprQoU01LrjrkOZk
DATA_PATH = "dinos.txt"


def one_hot_encode(array, vocab_size):
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    return np.eye(vocab_size, dtype=np.float32)[array]


class DinosDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        with open(data_path, "r", encoding="utf-8") as f:
            names = [line.strip().lower() for line in f if line.strip()]
        all_text = "".join(f"<{{name}}>" for name in names)
        self.vocab = sorted(set(all_text))
        self.vocab_size = len(self.vocab)
        self.id_to_token = {{i: ch for i, ch in enumerate(self.vocab)}}
        self.token_to_id = {{ch: i for i, ch in enumerate(self.vocab)}}
        self.corpus = []
        for name in names:
            self.corpus.extend([self.token_to_id[ch] for ch in f"<{{name}}>"])

    def __len__(self):
        return (len(self.corpus) - 1) // self.seq_len

    def __getitem__(self, index):
        s = index * self.seq_len
        return (
            torch.tensor(self.corpus[s : s + self.seq_len], dtype=torch.long),
            torch.tensor(self.corpus[s + 1 : s + self.seq_len + 1], dtype=torch.long),
        )


dataset = DinosDataset(DATA_PATH, SEQ_LEN)
vocab_size = dataset.vocab_size
n_total = len(dataset)
n_train = int(TRAIN_VAL_SPLIT * n_total)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, range(n_train)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, range(n_train, n_total)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
)


# ── Model ──
class CharRNN(nn.Module):
    def __init__(self, vocab_size, n_hidden, n_layers, drop_prob):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, n_hidden, n_layers,
                            dropout=drop_prob, batch_size=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0) * x.size(1), self.n_hidden)
        x = self.fc(x)
        return x, (h, c)

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        return (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
        )


model = CharRNN(vocab_size, N_HIDDEN, N_LAYERS, DROPOUT).to(device)
print(f"Parameters: {{sum(p.numel() for p in model.parameters()) / 1e6:.2f}}M")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss().to(device)


# ── Training ──
for epoch in range(EPOCHS):
    h = model.init_hidden(BATCH_SIZE, device)
    epoch_loss = 0
    n_batches = 0

    for x, targets in train_loader:
        x_enc = one_hot_encode(x, vocab_size)
        inputs = torch.from_numpy(x_enc).to(device)
        targets = targets.to(device)
        h = tuple(each.data for each in h)

        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output, targets.view(BATCH_SIZE * SEQ_LEN))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    # Validation
    model.train(False)
    val_h = model.init_hidden(BATCH_SIZE, device)
    val_losses = []
    for vx, vy in val_loader:
        vx_enc = one_hot_encode(vx, vocab_size)
        vx_t = torch.from_numpy(vx_enc).to(device)
        vy = vy.to(device)
        val_h = tuple(each.data for each in val_h)
        vout, val_h = model(vx_t, val_h)
        val_losses.append(criterion(vout, vy.view(BATCH_SIZE * SEQ_LEN)).item())
    model.train(True)

    print(f"Epoch {{epoch+1}}/{{EPOCHS}} | train {{epoch_loss/n_batches:.4f}} | val {{np.mean(val_losses):.4f}}")


# ── Generate ──
def sample(prefix="<", size=MAX_NEW_TOKENS):
    model.train(False)
    chars = list(prefix)
    h = model.init_hidden(1, device)
    for ch in prefix[:-1]:
        x = np.array([[dataset.token_to_id[ch]]])
        inp = torch.from_numpy(one_hot_encode(x, vocab_size)).float().to(device)
        with torch.no_grad():
            _, h = model(inp, h)
    last = prefix[-1]
    for _ in range(size):
        x = np.array([[dataset.token_to_id[last]]])
        inp = torch.from_numpy(one_hot_encode(x, vocab_size)).float().to(device)
        h = tuple(each.data for each in h)
        with torch.no_grad():
            out, h = model(inp, h)
            p = F.softmax(out, dim=1).cpu().numpy().squeeze()
        nxt = dataset.id_to_token[np.random.choice(len(dataset.id_to_token), p=p / p.sum())]
        chars.append(nxt)
        last = nxt
        if nxt == ">":
            break
    return "".join(chars)

print("\\n--- Generated names ---")
for _ in range(NUM_SAMPLES):
    print(sample())
'''


# ── Script dispatcher ──

SCRIPT_BUILDERS = {
    "transformer": _transformer_script,
    "rnn": _rnn_script,
}


def build_script(config: dict) -> str:
    """Build standalone .py script from config."""
    template = config.get("template", "transformer")
    builder = SCRIPT_BUILDERS.get(template)
    if builder is None:
        raise ValueError(f"Unknown template: {template}")
    return builder(config)


def build_notebook(config: dict) -> str:
    """Build .ipynb notebook from config. Returns JSON string."""
    script = build_script(config)
    nb = nbformat.v4.new_notebook()

    # Metadata cell
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# {config.get('name', 'LLM Experiments Lab Export')}\n\n"
        f"**Template:** {config.get('template', 'transformer')}\n\n"
        f"**Config:**\n```json\n{json.dumps(config, indent=2)}\n```\n\n"
        f"Exported from [LLM Experiments Lab]({settings.github_url})."
    ))

    # Split script into logical sections for separate cells
    sections = script.split("\n# ── ")
    # First section = imports + hyperparameters
    nb.cells.append(nbformat.v4.new_code_cell(sections[0]))
    # Remaining sections
    for section in sections[1:]:
        header_line = section.split("\n", 1)[0].rstrip(" ──")
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## {header_line}"))
        nb.cells.append(nbformat.v4.new_code_cell("# ── " + section))

    return nbformat.writes(nb)
