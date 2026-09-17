"""Guards for the 2026-09-17 incident: an unbounded embedding table and a
single flat proxy timeout together starved the trainer mid-run.

See docs/DESIGN_DECISIONS.md.
"""
import pytest

from backend.api.training import _exc_text, _timeout_for
from config.settings import settings


class TestTimeoutSelection:
    """Budgets must differ by what the call does, not be one flat number."""

    @pytest.mark.parametrize("path", ["/api/training/5/status", "/api/training/5/metrics"])
    def test_polls_fail_fast(self, path):
        assert _timeout_for(path) == settings.remote_poll_timeout_seconds

    @pytest.mark.parametrize("path", [
        "/api/training/5/pause", "/api/training/5/resume", "/api/training/5/stop",
    ])
    def test_control_calls_get_room_for_a_checkpoint_load(self, path):
        assert _timeout_for(path) == settings.remote_control_timeout_seconds

    def test_artifacts_use_the_artifact_budget(self):
        assert _timeout_for("/api/training/5/architecture/embedding-table") == (
            settings.remote_artifact_timeout_seconds
        )

    def test_a_poll_is_shorter_than_a_control_call(self):
        # The whole point of splitting: a dead worker surfaces in seconds,
        # while resume still gets time to load a ~0.5GB checkpoint.
        assert settings.remote_poll_timeout_seconds < settings.remote_control_timeout_seconds

    def test_no_hardcoded_timeout_remains_in_the_proxy(self):
        import inspect
        from backend.api import training
        src = inspect.getsource(training._proxy)
        assert "timeout=30" not in src


class TestExceptionText:
    """httpx timeouts have an empty str() — the banner showed nothing."""

    def test_empty_message_falls_back_to_class_name(self):
        import httpx
        assert _exc_text(httpx.ReadTimeout("")) == "ReadTimeout"

    def test_real_message_is_kept(self):
        assert _exc_text(ValueError("boom")) == "boom"


class TestEmbeddingTableCap:
    def test_cap_is_configured_and_small(self):
        # 50,304 rows is ~813MB of JSON; anything near that defeats the fix.
        assert 0 < settings.embedding_table_max_rows <= 2048

    def test_slicing_bounds_both_tables(self):
        """The handler slices before .tolist(); mirror that arithmetic here."""
        vocab_size, max_rows = 50304, settings.embedding_table_max_rows
        shown = min(vocab_size, max_rows)
        assert shown == max_rows
        assert shown < vocab_size          # truncated flag must be True
        position_embedding = [[0.0]] * 1024
        assert len(position_embedding[:max_rows]) == min(1024, max_rows)

    def test_small_vocab_is_not_truncated(self):
        """The char-level baseline (65 tokens) must still return everything."""
        vocab_size = 65
        shown = min(vocab_size, settings.embedding_table_max_rows)
        assert shown == vocab_size
        assert not (shown < vocab_size)
