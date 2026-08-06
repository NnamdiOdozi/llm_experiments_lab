"""Tests for BPE-aware bundle/zip export functionality."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from backend.api.codegen import export_bundle
from backend import db
from config.settings import settings


@pytest.mark.asyncio
async def test_char_export_bundle_no_artifact():
    """Char tokenizer export bundle should NOT include artifact."""
    # Find an experiment with char tokenizer or create one
    exp = await db.get_experiment(1)  # Should exist from prior tests
    if exp is None:
        pytest.skip("No experiment found for testing")

    config = json.loads(exp["config_json"])
    if config.get("data", {}).get("tokenizer", "char") != "char":
        pytest.skip("Test experiment is not char tokenizer")

    # Build bundle
    response = await export_bundle(1)
    buf = io.BytesIO(response.body)

    with zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()
        # Should have standard files
        assert "export.py" in names
        assert "export.ipynb" in names
        assert "config.json" in names
        assert "notes.md" in names

        # Should NOT have tokenizer artifacts
        json_files = [n for n in names if n.endswith(".json") and "tokenizer" in n]
        assert len(json_files) == 0, "Char export should not include tokenizer artifacts"


@pytest.mark.asyncio
async def test_bpe_export_bundle_includes_artifact():
    """BPE tokenizer export bundle SHOULD include artifact JSON."""
    # Find an experiment with BPE tokenizer
    for exp_id in range(1, 300):
        exp = await db.get_experiment(exp_id)
        if exp is None:
            continue

        config = json.loads(exp["config_json"])
        tokenizer_type = config.get("data", {}).get("tokenizer", "char")
        artifact = config.get("data", {}).get("tokenizer_artifact")

        if tokenizer_type != "char" and artifact:
            # Found a BPE experiment
            response = await export_bundle(exp_id)
            buf = io.BytesIO(response.body)

            with zipfile.ZipFile(buf, "r") as zf:
                names = zf.namelist()

                # Should have standard files
                assert "export.py" in names, f"Export missing export.py in bundle for exp {exp_id}"
                assert "export.ipynb" in names, f"Export missing export.ipynb in bundle for exp {exp_id}"
                assert "config.json" in names, f"Export missing config.json in bundle for exp {exp_id}"

                # Should have the tokenizer artifact
                assert artifact in names, f"Export missing artifact {artifact} in bundle for exp {exp_id}"

                # Verify artifact content is valid JSON
                artifact_content = zf.read(artifact).decode("utf-8")
                artifact_json = json.loads(artifact_content)
                assert "version" in artifact_json or "model" in artifact_json, "Artifact doesn't look like valid tokenizer"

            # Verify export.py script references the artifact correctly
            with zipfile.ZipFile(buf, "r") as zf:
                script = zf.read("export.py").decode("utf-8")
                assert f'Tokenizer.from_file("{artifact}")' in script, f"Script doesn't reference artifact {artifact}"
                assert "from tokenizers import Tokenizer" in script

            return  # Test passed

    pytest.skip("No BPE experiment found for testing")


@pytest.mark.asyncio
async def test_bpe_export_script_is_self_contained():
    """BPE export script in bundle should be self-contained with relative artifact path."""
    # Find a BPE experiment
    for exp_id in range(1, 300):
        exp = await db.get_experiment(exp_id)
        if exp is None:
            continue

        config = json.loads(exp["config_json"])
        tokenizer_type = config.get("data", {}).get("tokenizer", "char")
        artifact = config.get("data", {}).get("tokenizer_artifact")

        if tokenizer_type != "char" and artifact:
            response = await export_bundle(exp_id)
            buf = io.BytesIO(response.body)

            with zipfile.ZipFile(buf, "r") as zf:
                script = zf.read("export.py").decode("utf-8")

                # Should use relative path (just filename), not absolute path
                assert f'Tokenizer.from_file("{artifact}")' in script
                assert "/data/tokenizers/" not in script, "Script should use relative path, not absolute"
                assert "settings.data_dir" not in script or "Path" not in script, "Script should not reference settings"

            return  # Test passed

    pytest.skip("No BPE experiment found for testing")
