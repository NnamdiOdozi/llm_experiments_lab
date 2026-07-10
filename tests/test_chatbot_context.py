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


def test_get_last_audit_change_finds_matching_experiment(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-10 10:00:00 | INFO  | lab.audit | Experiment created: id=5 name='Test' preset=None\n"
        "2026-07-10 10:00:05 | INFO  | lab.audit | Config updated: experiment_id=5 changed={\"lr\": [0.001, 0.003]}\n"
        "2026-07-10 10:00:06 | INFO  | lab.audit | Config updated: experiment_id=50 changed={\"lr\": [0.001, 0.003]}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    result = context._get_last_audit_change(5)

    assert result is not None
    assert "experiment_id=5 " in result
    assert "lr" in result


def test_get_last_audit_change_no_match_returns_none(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text("2026-07-10 10:00:00 | INFO  | lab.audit | Experiment created: id=999 name='Other'\n")
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    assert context._get_last_audit_change(5) is None


def test_get_last_audit_change_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "get_log_path", lambda: tmp_path / "does_not_exist.log")
    assert context._get_last_audit_change(5) is None


def test_get_log_tail_returns_last_n_lines(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    tail = context._get_log_tail(5)

    assert len(tail) == 5
    assert tail[-1].strip() == "line 99"


def test_assemble_messages_structure(monkeypatch, tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-10 10:00:00 | INFO  | lab.audit | Config updated: experiment_id=1 changed={\"lr\": [0.001, 0.003]}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    experiment = {
        "id": 1,
        "name": "My experiment",
        "config_json": json.dumps({"template": "transformer", "description": "baseline"}),
    }
    run = {
        "status": "running",
        "current_step": 10,
        "total_steps": 100,
        "train_loss_history": json.dumps([{"step": 10, "train_loss": 2.0, "val_loss": 2.1}]),
    }
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    messages = context.assemble_messages(experiment, run, history, "What does this loss mean?")

    assert messages[0]["role"] == "system"
    assert "grounded lab assistant" in messages[0]["content"]
    assert messages[1]["role"] == "system"
    assert "RotaryPositionalEncoding" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "earlier question"}
    assert messages[3] == {"role": "assistant", "content": "earlier answer"}
    last = messages[-1]
    assert last["role"] == "user"
    assert "What does this loss mean?" in last["content"]
    assert "running" in last["content"]
    assert "experiment_id=1" in last["content"]
