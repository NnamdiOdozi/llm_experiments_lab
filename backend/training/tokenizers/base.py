"""Tokenizer protocol — unified interface for any tokenizer."""

from typing import Protocol


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers — any object implementing encode/decode/vocab_size.

    Allows different tokenizer implementations (char, BPE, etc.) to be swapped
    without runtime isinstance checks. Used with typing.Protocol for structural
    subtyping (duck typing).
    """

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary (max token ID + 1)."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Raw text to encode.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.
        """
        ...

    def id_to_token(self, token_id: int) -> dict:
        """Return metadata for a single token ID.

        Args:
            token_id: A token ID to inspect.

        Returns:
            Dict with keys:
            - "raw": The tokenizer's literal token string (bytes/subword unit).
            - "display": Human-safe compact form with byte-level markers visible
                         (e.g., Ġ→␠, newline→⏎). Used for UI/debugging.
            - "decoded": The result of decode([token_id]), i.e., the text this
                         single token represents.
        """
        ...
