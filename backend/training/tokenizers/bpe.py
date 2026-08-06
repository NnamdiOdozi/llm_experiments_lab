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
