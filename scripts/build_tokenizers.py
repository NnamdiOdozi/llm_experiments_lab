"""Build offline BPE tokenizer artifacts — deterministic, committable."""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

# Import data loader from the main codebase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.training.templates.transformer.data import load_tiny_shakespeare
from config.settings import settings


def train_bpe_tokenizer(text: str, vocab_size: int) -> Tokenizer:
    """Train a deterministic BPE tokenizer on text.

    Args:
        text: The corpus to train on.
        vocab_size: Target vocabulary size (may be slightly higher in practice).

    Returns:
        A trained tokenizers.Tokenizer instance ready to encode/decode.
    """
    # Create a tokenizer with byte-level BPE (deterministic)
    tokenizer = Tokenizer(BPE())

    # Set up normalization and pre-tokenization (byte-level)
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Train the tokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        # No randomness: fixed merges, deterministic order
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator([text], trainer=trainer)

    # Set the post-processor (byte-level encoding)
    tokenizer.post_processor = ByteLevelProcessor()

    # Set the decoder (reverse byte-level encoding for decode())
    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file for integrity checking.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the SHA256 hash.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    """Build offline BPE artifacts for Tiny Shakespeare."""
    # Create tokenizers directory
    tokenizers_dir = settings.data_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    # Load the corpus (cached if available)
    print("Loading Tiny Shakespeare corpus...")
    text = load_tiny_shakespeare()
    print(f"  Corpus size: {len(text):,} characters")

    # Build char tokenizer metadata (for reference)
    chars = sorted(set(text))
    char_meta = {
        "id": "char",
        "vocab_size": len(chars),
        "version": "v1",
    }

    # Train BPE tokenizers
    bpe_configs = [
        ("1k", 1024),
        ("4k", 4096),
    ]

    manifest = {"tokenizers": [char_meta]}

    for variant, target_vocab in bpe_configs:
        print(f"\nTraining BPE {variant} (target vocab_size={target_vocab})...")
        tokenizer = train_bpe_tokenizer(text, target_vocab)

        # Save the tokenizer
        artifact_name = f"tiny-shakespeare-bpe-{variant}-v1.json"
        artifact_path = tokenizers_dir / artifact_name
        tokenizer.save(str(artifact_path))

        # Verify by reloading
        reloaded = Tokenizer.from_file(str(artifact_path))
        actual_vocab_size = reloaded.get_vocab_size()

        # Compute hash
        file_hash = compute_file_hash(artifact_path)
        file_size = artifact_path.stat().st_size

        meta = {
            "id": f"bpe_{variant}",
            "variant": variant,
            "target_vocab_size": target_vocab,
            "actual_vocab_size": actual_vocab_size,
            "sha256": file_hash,
            "file_size": file_size,
            "version": "v1",
        }

        manifest["tokenizers"].append(meta)

        print(f"  ✓ Saved to {artifact_name}")
        print(f"    Target vocab_size: {target_vocab}")
        print(f"    Actual vocab_size: {actual_vocab_size}")
        print(f"    SHA256: {file_hash[:16]}...")
        print(f"    File size: {file_size:,} bytes")

    # Write manifest
    manifest_path = tokenizers_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Verify reproducibility by running a second time
    print("\n" + "=" * 60)
    print("Verifying determinism (rebuilding artifacts)...")
    print("=" * 60)

    for variant, target_vocab in bpe_configs:
        print(f"\nRebuild {variant}...")
        tokenizer = train_bpe_tokenizer(text, target_vocab)
        artifact_name = f"tiny-shakespeare-bpe-{variant}-v1.json"
        temp_path = tokenizers_dir / f".{artifact_name}.tmp"
        tokenizer.save(str(temp_path))

        artifact_path = tokenizers_dir / artifact_name

        hash1 = compute_file_hash(temp_path)
        hash2 = compute_file_hash(artifact_path)

        if hash1 == hash2:
            print(f"  ✓ DETERMINISTIC: hashes match")
        else:
            print(f"  ✗ MISMATCH: {hash1[:16]}... vs {hash2[:16]}...")

        temp_path.unlink()

    print("\n✓ Build complete!")
    print(f"\nArtifacts ready in {tokenizers_dir}")
    print("Commit the following files:")
    for variant in ["1k", "4k"]:
        artifact_name = f"tiny-shakespeare-bpe-{variant}-v1.json"
        print(f"  - data/tokenizers/{artifact_name}")
    print(f"  - data/tokenizers/manifest.json")


if __name__ == "__main__":
    main()
