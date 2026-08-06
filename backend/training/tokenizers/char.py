"""Character-level tokenizer — wraps existing char-level encoding logic."""


class CharTokenizer:
    """Character-level tokenizer: one token per character."""

    def __init__(self, chars: list[str]):
        """Initialize from an explicit sorted list of characters.

        Args:
            chars: Sorted list of unique characters (vocabulary).
        """
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a CharTokenizer from raw text, inferring vocabulary.

        Args:
            text: Text to infer character vocabulary from.

        Returns:
            A CharTokenizer instance with sorted unique characters from text.
        """
        chars = sorted(set(text))
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Encode text to character IDs.

        Args:
            text: Text to encode.

        Returns:
            List of character token IDs.
        """
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode character IDs back to text.

        Args:
            ids: List of character token IDs.

        Returns:
            Decoded text.
        """
        return "".join(self.itos[int(i)] for i in ids)

    def id_to_token(self, token_id: int) -> dict:
        """Return metadata for a character token.

        Args:
            token_id: A character token ID.

        Returns:
            Dict with raw/display/decoded fields. For characters, all three
            are the character itself (with whitespace made visible in display).
        """
        ch = self.itos[token_id]

        # Display version: make whitespace visible
        if ch == "\n":
            display = "⏎"
        elif ch == " ":
            display = "␠"
        elif ch == "\t":
            display = "␉"
        else:
            display = ch

        return {
            "raw": ch,
            "display": display,
            "decoded": ch,
        }
