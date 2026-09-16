"""The controller must notice when the trainer image runs different source.

Real incident, 2026-09-16: gradient clipping was added to train_worker.py,
which runs INSIDE the Nebius container. Until the image is rebuilt, the
container has no code to read the new grad_clip key, so it is silently
ignored — the run trains normally and nothing anywhere reports a problem.

These tests pin the warning, not the training behaviour: a stale image must
never block a run, only shout about it.
"""

import httpx
import pytest

from backend.api import training as training_api
from backend.provenance import source_fingerprint


def test_fingerprint_is_stable_and_short():
    a, b = source_fingerprint(), source_fingerprint()
    assert a == b, "fingerprint must be deterministic across calls"
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


def test_fingerprint_changes_when_a_tracked_file_changes(tmp_path, monkeypatch):
    """A content change under backend/ must move the fingerprint."""
    from backend import provenance

    before = provenance.source_fingerprint()
    target = provenance._ROOT / "backend" / "provenance.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# temporary edit for test\n")
        assert provenance.source_fingerprint() != before
    finally:
        target.write_bytes(original)
    assert provenance.source_fingerprint() == before, "test failed to restore the file"


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _Client:
    def __init__(self, payload=None, exc=None): self._p, self._exc = payload, exc
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url):
        if self._exc: raise self._exc
        return _Resp(self._p)


def _patch_client(monkeypatch, **kw):
    monkeypatch.setattr(training_api.httpx, "AsyncClient", lambda *a, **k: _Client(**kw))


@pytest.mark.asyncio
async def test_matching_fingerprint_logs_info_not_warning(monkeypatch, caplog):
    _patch_client(monkeypatch, payload={"source_fingerprint": source_fingerprint()})
    with caplog.at_level("INFO"):
        await training_api._warn_if_image_is_stale("http://endpoint", 1)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.asyncio
async def test_mismatched_fingerprint_warns(monkeypatch, caplog):
    _patch_client(monkeypatch, payload={"source_fingerprint": "deadbeefcafe"})
    with caplog.at_level("WARNING"):
        await training_api._warn_if_image_is_stale("http://endpoint", 7)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("STALE TRAINER IMAGE" in m for m in msgs), msgs
    assert any("build_push_trainer_gpu.sh" in m for m in msgs), "warning must say how to fix it"


@pytest.mark.asyncio
async def test_endpoint_predating_the_check_warns(monkeypatch, caplog):
    """An old image has no source_fingerprint field at all — that is itself stale."""
    _patch_client(monkeypatch, payload={"status": "ok"})
    with caplog.at_level("WARNING"):
        await training_api._warn_if_image_is_stale("http://endpoint", 8)
    assert any("STALE TRAINER IMAGE" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")


@pytest.mark.asyncio
async def test_unreachable_health_warns_but_does_not_raise(monkeypatch, caplog):
    """Never block a run just because the health probe failed."""
    _patch_client(monkeypatch, exc=httpx.ConnectError("boom"))
    with caplog.at_level("WARNING"):
        await training_api._warn_if_image_is_stale("http://endpoint", 9)   # must not raise
    assert any("Could not read remote source fingerprint" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")


def test_fingerprint_skips_symlinks_so_host_and_container_agree():
    """backend/training/templates/moe/data.py is a symlink to an ABSOLUTE host
    path, so it dangles inside the trainer container. Reading it raised
    FileNotFoundError and would have crashed /api/health in the trainer.
    Following symlinks would also make the two sides disagree forever, flagging
    every image as stale. Both sides must skip them."""
    from backend import provenance

    tracked = provenance._tracked_files()
    assert tracked, "fingerprint tracks no files at all"
    assert not [p for p in tracked if p.is_symlink()], "symlinks must be excluded"
    # and it must not raise on this repo, which contains exactly such a symlink
    assert provenance.source_fingerprint()
