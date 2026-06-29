"""CharRNN language model — extracted from RNN_LM_homework notebook."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode(array, vocab_size: int) -> np.ndarray:
    """One-hot encode an array of token indices."""
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    return np.eye(vocab_size, dtype=np.float32)[array]


class CharRNN(nn.Module):
    """Character-level RNN (LSTM) language model."""

    def __init__(
        self,
        vocab_size: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(
            vocab_size, n_hidden, n_layers,
            dropout=drop_prob, batch_first=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0) * x.size(1), self.n_hidden)
        x = self.fc(x)
        return x, (h, c)

    def init_hidden(self, batch_size: int, device: str = "cpu"):
        """Initialize hidden state to zeros."""
        weight = next(self.parameters()).data
        return (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
        )

    @torch.no_grad()
    def generate(self, id_to_token, token_to_id, prefix="<", max_new_tokens=100, device="cpu", temperature=0.8):
        """Generate text character by character from a prefix."""
        self.to(device)
        self.set_eval_mode()

        chars = list(prefix)
        h = self.init_hidden(1, device)

        # Feed prefix through model
        for ch in prefix[:-1]:
            x = np.array([[token_to_id[ch]]])
            inputs = one_hot_encode(x, len(token_to_id))
            inputs = torch.from_numpy(inputs).float().to(device)
            _, h = self(inputs, h)

        last_char = prefix[-1]

        for _ in range(max_new_tokens):
            x = np.array([[token_to_id[last_char]]])
            inputs = one_hot_encode(x, len(token_to_id))
            inputs = torch.from_numpy(inputs).float().to(device)
            h = tuple(each.data for each in h)

            out, h = self(inputs, h)
            # Scale logits by temperature before softmax — controls randomness
            p = F.softmax(out / temperature, dim=1).cpu().numpy().squeeze()

            next_char_id = np.random.choice(len(token_to_id), p=p / p.sum())
            next_char = id_to_token[next_char_id]
            chars.append(next_char)
            last_char = next_char

            if next_char == ">":
                break

        return "".join(chars)

    def set_eval_mode(self):
        """Switch model to evaluation mode."""
        self.train(False)


def build_model_from_config(config: dict) -> CharRNN:
    """Instantiate a CharRNN from an experiment config dict."""
    m = config["model"]
    return CharRNN(
        vocab_size=m["vocab_size"],
        n_hidden=m.get("n_hidden", 256),
        n_layers=m.get("n_layers", 2),
        drop_prob=m.get("dropout", 0.5),
    )
