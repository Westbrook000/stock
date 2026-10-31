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
    assert not (tmp_path / "data/market_data_versions/latest-refresh.json").exists()


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




def test_short_abnormal_provider_history_is_accepted_without_old_file_hashes(tmp_path):
    path=seed_file(tmp_path)
    rows=provider_frame(2).assign(code="999999")
    rows.loc[0,"high"]=1
    rows.loc[1,"volume"]=-50
    rows["date"]=pd.bdate_range("2026-06-01",periods=2)
    result=refresh_one(tmp_path,"000001","stock",provider=lambda *args:rows)
    written=pd.read_csv(path,encoding="utf-8-sig",dtype={"code":str})
    assert result["rows"]==2
    assert written.high.iloc[0]==1
    assert written.volume.iloc[1]==-50
    assert written.code.tolist()==["000001","000001"]
    assert result["first_date"]=="2026-06-01"
    assert not ({"file_sha256","previous_sha256","previous_snapshot"}&result.keys())
    assert not (tmp_path/"data/market_data_versions").exists()


def test_provider_and_normalization_failure_keep_existing_csv(tmp_path):
    path=seed_file(tmp_path);original=path.read_bytes();calls=[]
    def provider(*args):
        calls.append(args)
        raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError,match="provider unavailable"):
        refresh_one(tmp_path,"000001","stock",provider=provider)
    assert len(calls)==1
    assert path.read_bytes()==original
    with pytest.raises(KeyError):
        refresh_one(tmp_path,"000001","stock",provider=lambda *args:pd.DataFrame())
    assert path.read_bytes()==original


def test_basis_write_failure_keeps_existing_csv(tmp_path,monkeypatch):
    import caisen.data_refresh as module
    path=seed_file(tmp_path);original=path.read_bytes()
    original_write=module._atomic_write
    def fail(target,content):
        if target.name==".market-data-basis.json": raise OSError("write failed")
        return original_write(target,content)
    monkeypatch.setattr(module,"_atomic_write",fail)
    with pytest.raises(OSError,match="write failed"):
        refresh_one(tmp_path,"000001","stock",provider=lambda *args:provider_frame())
    assert path.read_bytes()==original
