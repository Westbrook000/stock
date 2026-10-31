import os

import pytest


@pytest.fixture(autouse=True)
def isolate_caisen_test_state(tmp_path, monkeypatch):
    """Keep every test's SQLite writes and API job caches away from user data."""
    monkeypatch.setenv("CAISEN_DB", os.path.join(str(tmp_path), "caisen-tests.sqlite3"))
    try:
        from caisen import api
    except ImportError:
        api = None
    if api is not None:
        api._jobs.clear()
        api._fixed_backtest_jobs.clear()
    yield
    if api is not None:
        api._jobs.clear()
        api._fixed_backtest_jobs.clear()
