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


def test_get_recent_training_events_matches_lines_with_no_trailing_content(monkeypatch, tmp_path):
    """Several lab.training messages (STOP, PAUSE requested) end right after
    the run_id digits with nothing after — a trailing-space marker would
    miss them entirely."""
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-12 15:35:33 | INFO  | lab.training | PAUSE requested run_id=1204\n"
        "2026-07-12 15:40:00 | INFO  | lab.training | RESUME run_id=1204 config_refreshed=True\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    events = context._get_recent_training_events(1204, 10)

    assert len(events) == 2
    assert "PAUSE requested run_id=1204" in events[0]
    assert "RESUME run_id=1204" in events[1]


def test_get_recent_training_events_does_not_false_match_prefix_run_id(monkeypatch, tmp_path):
    """run_id=5 must not match a line about run_id=50."""
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-12 15:35:33 | INFO  | lab.training | STOP run_id=50\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    events = context._get_recent_training_events(5, 10)

    assert events == []


def test_get_recent_errors_filters_by_category(monkeypatch, tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-12 09:00:00 | INFO  | lab.request | GET /api/experiments\n"
        "2026-07-12 09:00:01 | ERROR | lab.error | Something broke: run_id=5\n"
        "2026-07-12 09:00:02 | INFO  | lab.training | Run 5 started\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    errors = context._get_recent_errors(10)

    assert len(errors) == 1
    assert "Something broke" in errors[0]


def test_format_resource_usage_with_no_run_id():
    assert context._format_resource_usage(None) is None


def test_format_resource_usage_summarizes_latest_sample(monkeypatch):
    monkeypatch.setattr(
        context,
        "read_metrics_from_disk",
        lambda run_id: [
            {"step": 1, "cpu_percent": 10.0},
            {"step": 2, "cpu_percent": 42.0, "gpu_utilization_pct": 87.0, "gpu_temp_c": 65.0},
        ],
    )

    summary = context._format_resource_usage(run_id=1)

    assert summary is not None
    assert "step 2" in summary
    assert "CPU 42%" in summary
    assert "GPU 87%" in summary
    assert "GPU temp 65C" in summary


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


def test_get_last_audit_change_skips_trailing_noop_diff(tmp_path, monkeypatch):
    """Regression test for a real live incident (2026-07-13): a debounced
    autosave logging an empty diff as the literal last audit line buried a
    real config change, and the chatbot told the user 'no config
    modifications have been made' right after they'd changed eval_interval
    20->10."""
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-13 15:36:19 | INFO  | lab.audit | Config updated: experiment_id=148 changed={\"eval_interval\": [20, 10]}\n"
        "2026-07-13 15:36:27 | INFO  | lab.audit | Config updated: experiment_id=148 changed={}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    result = context._get_last_audit_change(148)

    assert result is not None
    assert "eval_interval" in result


def test_get_last_audit_change_skips_non_config_lines(tmp_path, monkeypatch):
    """A 'Notes updated' line matches the same id=<N> marker but isn't a
    config change — must not win over a real config change just for being
    more recent in the log."""
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-13 15:36:19 | INFO  | lab.audit | Config updated: experiment_id=148 changed={\"eval_interval\": [20, 10]}\n"
        "2026-07-13 16:12:03 | INFO  | lab.audit | Notes updated: experiment_id=148 len=193\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    result = context._get_last_audit_change(148)

    assert result is not None
    assert "eval_interval" in result


def test_get_last_audit_change_returns_none_when_only_noop_diffs_exist(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-13 15:36:19 | INFO  | lab.audit | Config updated: experiment_id=148 changed={}\n"
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    assert context._get_last_audit_change(148) is None


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
    assert "Project README" in messages[1]["content"]
    assert messages[2]["role"] == "system"
    assert "RotaryPositionalEncoding" in messages[2]["content"]
    assert messages[3] == {"role": "user", "content": "earlier question"}
    assert messages[4] == {"role": "assistant", "content": "earlier answer"}
    last = messages[-1]
    assert last["role"] == "user"
    assert "What does this loss mean?" in last["content"]
    assert "running" in last["content"]
    assert "experiment_id=1" in last["content"]


def _prompt_line(run_id: int, step: int, prompt: str, output: str) -> str:
    payload = json.dumps({"step": step, "prompt": prompt, "output": output})
    return f"2026-07-11 12:00:00 | INFO  | lab.prompt | run_id={run_id} payload={payload}\n"


def test_get_prompt_history_returns_pairs_for_run(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        _prompt_line(7, 500, "The king said", "xqz jkl")
        + _prompt_line(99, 500, "other run", "ignored")
        + _prompt_line(7, 3000, "The king said", "to the queen, well")
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    history = context._get_prompt_history(7)

    assert len(history) == 2
    assert history[0] == {"step": 500, "prompt": "The king said", "output": "xqz jkl"}
    assert history[1]["step"] == 3000
    assert "queen" in history[1]["output"]


def test_get_prompt_history_skips_malformed_and_missing_file(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        "2026-07-11 12:00:00 | INFO  | lab.prompt | run_id=7 payload={broken json\n"
        + _prompt_line(7, 100, "ok", "fine")
    )
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)
    assert context._get_prompt_history(7) == [{"step": 100, "prompt": "ok", "output": "fine"}]

    monkeypatch.setattr(context, "get_log_path", lambda: tmp_path / "nope.log")
    assert context._get_prompt_history(7) == []


def test_volatile_snapshot_includes_prompt_history(tmp_path, monkeypatch):
    log_file = tmp_path / "session.log"
    log_file.write_text(_prompt_line(7, 500, "The king said", "xqz jkl"))
    monkeypatch.setattr(context, "get_log_path", lambda: log_file)

    run = {"id": 7, "status": "paused", "current_step": 500, "total_steps": 1000,
           "train_loss_history": json.dumps([])}
    snapshot = context._build_volatile_snapshot(1, run)

    assert "The king said" in snapshot
    assert "xqz jkl" in snapshot
    assert "step 500" in snapshot
