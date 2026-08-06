"""Load tokenizer manifest for metadata (vocab_size, hash, version)."""

import json
from pathlib import Path


def load_manifest() -> dict:
    """Load tokenizer manifest.json.

    Returns:
        Dict with "tokenizers" list containing metadata for each tokenizer.
    """
    manifest_path = Path(__file__).parent / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)
