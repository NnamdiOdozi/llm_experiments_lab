import json
from backend.chatbot import context


def test_read_template_source_includes_real_transformer_code():
    source = context._read_template_source("transformer")
    assert "class RotaryPositionalEncoding" in source
    assert "model.py" in source
    assert "data.py" in source


def test_read_template_source_unknown_template_does_not_crash():
    source = context._read_template_source("nonexistent_template")
    assert "No source found" in source


def test_format_loss_snapshot_with_no_run():
    snapshot = context._format_loss_snapshot(None)
    assert "No training run" in snapshot


def test_format_loss_snapshot_with_run():
    run = {
        "status": "running",
        "current_step": 40,
        "total_steps": 1000,
        "train_loss_history": json.dumps([
            {"step": 20, "train_loss": 1.8, "val_loss": 1.9},
            {"step": 40, "train_loss": 1.5, "val_loss": 1.6},
        ]),
    }
    snapshot = context._format_loss_snapshot(run)
    assert "running" in snapshot
    assert "40 / 1000" in snapshot
    assert "1.5" in snapshot
