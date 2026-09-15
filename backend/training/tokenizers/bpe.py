"""Byte-Pair Encoding tokenizer — wraps Hugging Face tokenizers library."""

from pathlib import Path
from tokenizers import Tokenizer


class BPETokenizer:
    """BPE tokenizer using Hugging Face tokenizers library."""

    def __init__(self, tokenizer_path: str | Path):
        """Initialize from a saved HF tokenizers JSON artifact.

        Args:
            tokenizer_path: Path to the .json file (from tokenizer.save()).
        """
        self.tokenizer_path = Path(tokenizer_path)
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    @property
    def vocab_size(self) -> int:
        """Return the size of the BPE vocabulary."""
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of BPE token IDs.
        """
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode BPE token IDs back to text.

        Args:
            ids: List of BPE token IDs.

        Returns:
            Decoded text.
        """
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def id_to_token(self, token_id: int) -> dict:
        """Return metadata for a BPE token.

        Args:
            token_id: A BPE token ID to inspect.

        Returns:
            Dict with raw/display/decoded fields. Display makes byte-level
            markers visible (e.g., Ġ→␠, Ċ→⏎) for readability.
        """
        # Get the raw token string from the tokenizer's vocab
        raw = self._tokenizer.id_to_token(token_id)

        # Make byte-level markers visible in the display version
        display = raw
        display = display.replace("Ġ", "␠")  # HF byte-level BPE space marker
        display = display.replace("Ċ", "⏎")  # HF byte-level BPE newline marker

        # Decode just this token
        decoded = self.decode([token_id])

        return {
            "raw": raw,
            "display": display,
            "decoded": decoded,
        }


# Karpathy's nanoGPT rounds the real GPT-2 BPE vocab (50257) up to the next
# multiple of 64 = 50304, purely for GPU memory alignment. The model's
# embedding/lm_head are sized 50304; the 47 extra rows (50257..50303) are
# never trained. Reporting 50304 here is what makes the nanoGPT preset reach
# the full 124.4M-param count. See config/presets.py.
GPT2_VOCAB = 50304
GPT2_REAL_VOCAB = 50257


class Gpt2Tokenizer(BPETokenizer):
    """The pretrained GPT-2 BPE tokenizer (openai-community/gpt2 artifact).

    Reuses BPETokenizer for encode/decode/id_to_token, but reports a padded
    vocab_size of 50304 (GPT-2 rounded up to a multiple of 64) so the model
    embedding matches nanoGPT's 124.4M-param configuration. Encode never
    emits an id >= 50257 (the real GPT-2 vocab), so the padding rows are
    simply never selected.
    """

    def __init__(self, tokenizer_path: str | Path):
        super().__init__(tokenizer_path)
        self._real_vocab = super().vocab_size  # 50257 from the artifact

    @property
    def vocab_size(self) -> int:
        return GPT2_VOCAB

    def id_to_token(self, token_id: int) -> dict:
        if token_id >= self._real_vocab:
            # Padding row — no real token exists. Return a placeholder instead
            # of crashing on an out-of-range lookup.
            return {
                "raw": f"<pad:{token_id}>",
                "display": f"<pad:{token_id}>",
                "decoded": "",
            }
        return super().id_to_token(token_id)
