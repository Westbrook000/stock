import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from caisen.data_refresh import refresh_one
from caisen.market import InstrumentFile, load_daily_bars


def seed_file(root, kind="stock", code="000001"):
    folder = root / "data" / ("stocks" if kind == "stock" else "etfs")
    folder.mkdir(parents=True)
    path = folder / f"{code}.csv"
    path.write_text("date,code,name,open,high,low,close,volume\n2026-01-02,000001,旧名称,10,11,9,10,100\n", encoding="utf-8")
    return path


def provider_frame(rows=24):
    dates = pd.bdate_range("2026-01-02", periods=rows)
    values = [10 + i / 100 for i in range(rows)]
    return pd.DataFrame({"date": dates, "open": values, "high": [v + .1 for v in values],
                         "low": [v - .1 for v in values], "close": values, "volume": [1000] * rows,
                         "name": ["新名称"] * rows})


def test_etf_provider_symbol_includes_exchange_prefix(monkeypatch):
    import sys
    from types import SimpleNamespace
    from caisen.data_refresh import fetch_history
    requested = []
    provider = SimpleNamespace(fund_etf_hist_sina=lambda symbol: requested.append(symbol) or provider_frame())
    monkeypatch.setitem(sys.modules, "akshare", provider)

    fetch_history("510300", "etf")
    fetch_history("159919", "etf")

    assert requested == ["sh510300", "sz159919"]


def test_refresh_replaces_csv_atomically_and_records_hash_provenance(tmp_path):
    path = seed_file(tmp_path)
    old_bytes = path.read_bytes()
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    result = refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame().assign(code="000001"))

    assert result["previous_sha256"] == old_hash
    assert result["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert Path(result["previous_snapshot"]).read_bytes() == old_bytes
    assert result["last_date"] == str(provider_frame().date.iloc[-1].date())
    assert result["adjustment"] == "qfq"
    assert result["provider_attempts"] == 1
    assert pd.read_csv(path, encoding="utf-8-sig").shape[0] == 24
    manifest = (tmp_path / "data/market_data_versions/latest-refresh.json").read_text(encoding="utf-8")
    assert '"stock:000001"' in manifest


def test_failed_or_too_short_provider_response_keeps_existing_file(tmp_path):
    path = seed_file(tmp_path)
    old_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="少于20"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame(5))
    assert path.read_bytes() == old_bytes
    calls = []
    def deterministic_provider(code, kind):
        calls.append((code, kind))
        raise RuntimeError("provider down")
    with pytest.raises(RuntimeError, match="provider down"):
        refresh_one(tmp_path, "000001", "stock", provider=deterministic_provider)
    assert len(calls) == 1
    assert path.read_bytes() == old_bytes


def test_refresh_retries_provider_rate_limit_with_exponential_backoff(tmp_path, monkeypatch):
    import caisen.data_refresh as refresh_module

    path = seed_file(tmp_path)
    calls = []
    sleeps = []

    class RateLimitError(RuntimeError):
        response = type("Response", (), {"status_code": 429})()

    def provider(code, kind):
        calls.append((code, kind))
        if len(calls) < 3:
            raise RateLimitError("rate limited")
        return provider_frame()

    monkeypatch.setattr(refresh_module.time, "sleep", sleeps.append)
    result = refresh_module.refresh_one(tmp_path, "000001", "stock", provider=provider)

    assert result["rows"] == 24
    assert result["provider_attempts"] == 3
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert pd.read_csv(path, encoding="utf-8-sig").shape[0] == 24
    manifest = json.loads((tmp_path / "data/market_data_versions/latest-refresh.json").read_text(encoding="utf-8"))
    assert manifest["symbols"]["stock:000001"]["provider_attempts"] == 3


def test_transient_provider_failures_stop_after_retry_limit_and_keep_file(tmp_path, monkeypatch):
    import caisen.data_refresh as refresh_module

    path = seed_file(tmp_path)
    original = path.read_bytes()
    calls = []
    sleeps = []

    class TimeoutErrorFromProvider(TimeoutError):
        pass

    def provider(code, kind):
        calls.append((code, kind))
        raise TimeoutErrorFromProvider("read timed out")

    monkeypatch.setattr(refresh_module.time, "sleep", sleeps.append)
    with pytest.raises(refresh_module.ProviderRetriesExhausted, match="3 次") as raised:
        refresh_module.refresh_one(tmp_path, "000001", "stock", provider=provider)

    assert raised.value.attempts == 3
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert path.read_bytes() == original
    assert not (tmp_path / "data/market_data_versions/latest-refresh.json").exists()


def test_invalid_ohlcv_rows_are_removed_and_all_invalid_response_is_rejected(tmp_path):
    path = seed_file(tmp_path)
    rows = provider_frame()
    rows.loc[2, "high"] = 1
    result = refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: rows)
    assert result["rows"] == 23
    old = path.read_bytes()
    with pytest.raises(ValueError, match="没有通过"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame().assign(volume=-1))
    assert path.read_bytes() == old


def test_empty_provider_response_keeps_local_history(tmp_path):
    path = seed_file(tmp_path)
    old = path.read_bytes()
    with pytest.raises(ValueError, match="行情源返回空数据"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: pd.DataFrame())
    assert path.read_bytes() == old


def test_provider_symbol_mismatch_is_rejected_before_snapshot_or_manifest(tmp_path):
    path = seed_file(tmp_path)
    original = path.read_bytes()
    rows = provider_frame().assign(code="000002")

    with pytest.raises(ValueError, match="代码与请求标的不一致"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: rows)

    assert path.read_bytes() == original
    assert not list((tmp_path / "data/market_data_versions").rglob("*.csv"))
    assert not (tmp_path / "data/market_data_versions/latest-refresh.json").exists()
    assert not (tmp_path / "data/market_data_current/.market-data-basis.json").exists()


def test_refresh_manifest_top_level_timestamp_tracks_latest_symbol_refresh(tmp_path):
    path = seed_file(tmp_path)
    versions_dir = tmp_path / "data" / "market_data_versions"
    versions_dir.mkdir(parents=True)
    manifest_path = versions_dir / "latest-refresh.json"
    manifest_path.write_text(json.dumps({"updated_at": "2020-01-01T00:00:00+08:00", "symbols": {}}), encoding="utf-8")

    refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["updated_at"] != "2020-01-01T00:00:00+08:00"
    assert manifest["updated_at"] == manifest["symbols"]["stock:000001"]["updated_at"]


def test_incomplete_provider_history_is_rejected_without_changing_local_file(tmp_path):
    path = seed_file(tmp_path)
    original = path.read_bytes()
    rows = provider_frame()
    rows = rows.loc[rows.date > pd.Timestamp("2026-01-02")].copy()

    with pytest.raises(ValueError, match="未覆盖本地历史"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: rows)

    assert path.read_bytes() == original
    assert not (tmp_path / "data/market_data_versions/latest-refresh.json").exists()
    assert not list((tmp_path / "data/market_data_versions").rglob("*.csv"))


def test_provider_must_include_every_existing_bar_date_not_only_endpoints(tmp_path):
    path = seed_file(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("2026-01-05,000001,旧名称,10,11,9,10,100\n")
    original = path.read_bytes()
    rows = provider_frame()
    rows = rows.loc[rows.date != pd.Timestamp("2026-01-05")].copy()

    with pytest.raises(ValueError, match="未覆盖本地历史"):
        refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: rows)

    assert path.read_bytes() == original


def test_refreshed_data_exposes_symbol_provenance_only_when_hash_matches(tmp_path):
    path = seed_file(tmp_path)
    refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame())
    result = load_daily_bars(InstrumentFile("000001", "stock", str(path)), root=tmp_path)
    assert result.bars.attrs["provenance"]["refresh"]["source"] == "AkShare Sina daily history"

    path.write_text(path.read_text(encoding="utf-8-sig") + "2026-10-20,000001,extra,10,11,9,10,100\n", encoding="utf-8-sig")
    changed = load_daily_bars(InstrumentFile("000001", "stock", str(path)), root=tmp_path)
    assert changed.bars.attrs["provenance"]["refresh"] is None


def test_failed_final_atomic_swap_does_not_replace_market_file(tmp_path, monkeypatch):
    import caisen.data_refresh as refresh_module
    path = seed_file(tmp_path)
    original = path.read_bytes()
    atomic_write = refresh_module._atomic_write
    def fail_market_csv(target, content):
        if target == path:
            raise OSError("disk full")
        return atomic_write(target, content)
    monkeypatch.setattr(refresh_module, "_atomic_write", fail_market_csv)
    with pytest.raises(OSError, match="disk full"):
        refresh_module.refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame())
    assert path.read_bytes() == original


def test_metadata_commit_failure_rolls_back_market_and_existing_metadata(tmp_path, monkeypatch):
    import caisen.data_refresh as refresh_module
    path = seed_file(tmp_path)
    original_market = path.read_bytes()
    current_dir = tmp_path / "data" / "market_data_current"
    current_dir.mkdir(parents=True)
    basis_path = current_dir / ".market-data-basis.json"
    basis_path.write_text('{"keep":"basis"}', encoding="utf-8")
    versions_dir = tmp_path / "data" / "market_data_versions"
    versions_dir.mkdir(parents=True)
    manifest_path = versions_dir / "latest-refresh.json"
    manifest_path.write_text('{"keep":"manifest"}', encoding="utf-8")
    original_basis, original_manifest = basis_path.read_bytes(), manifest_path.read_bytes()

    atomic_write = refresh_module._atomic_write
    failed = False
    def fail_first_manifest_write(target, content):
        nonlocal failed
        if target == manifest_path and not failed:
            failed = True
            raise OSError("manifest storage failed")
        return atomic_write(target, content)
    monkeypatch.setattr(refresh_module, "_atomic_write", fail_first_manifest_write)

    with pytest.raises(OSError, match="manifest storage failed"):
        refresh_module.refresh_one(tmp_path, "000001", "stock", provider=lambda code, kind: provider_frame())
    assert path.read_bytes() == original_market
    assert basis_path.read_bytes() == original_basis
    assert manifest_path.read_bytes() == original_manifest
