"""Shared pytest fixtures for the treasury-agent test suite."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Make the treasury-agent package importable when running pytest from
# any working directory (e.g. repo root or the treasury-agent/ dir).
_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# FX rates fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=False)
def fx_clean_cache(monkeypatch: pytest.MonkeyPatch):
    """Redirect the FX disk-cache to a /tmp dir and wipe in-memory state.

    Uses tempfile.mkdtemp() under /tmp so that cleanup works correctly in
    all filesystem environments (including network/FUSE mounts).
    """
    from fx import rates  # noqa: PLC0415

    tmp_dir = Path(tempfile.mkdtemp(prefix="treasury-test-fx-"))
    fake_cache = tmp_dir / "fx_rates.json"
    monkeypatch.setattr(rates, "_CACHE_PATH", fake_cache)
    monkeypatch.setattr(rates, "_cache", {})
    yield fake_cache
    # Reset after test so the module-level dict is clean for the next test
    monkeypatch.setattr(rates, "_cache", {})
