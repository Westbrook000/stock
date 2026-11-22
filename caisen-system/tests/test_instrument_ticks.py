import json

import pytest

from caisen.daily_research import _instrument_tick, _signal_cache_identity, load_research_bars
from caisen.market import InstrumentFile, load_daily_bars, resolve_instrument_tick


def _write_bars(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,100\n")


def test_kind_defaults_match_exchange_price_increment_and_registry_overrides(tmp_path):
    registry = tmp_path / "caisen-system/config/instruments.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"instruments": {
        "etf:510050": {"tick_size": 0.005},
        "stock:600000": {"exchange_asset_class": "stock"},
    }}))
    etf = InstrumentFile("510051", "etf", str(tmp_path / "data/etfs/510051.csv"))
    stock = InstrumentFile("600001", "stock", str(tmp_path / "data/stocks/600001.csv"))
    override = InstrumentFile("510050", "etf", str(tmp_path / "data/etfs/510050.csv"))
    _write_bars(tmp_path / etf.path)
    _write_bars(tmp_path / stock.path)
    _write_bars(tmp_path / override.path)

    assert _instrument_tick(tmp_path, etf) == (0.001, "exchange_asset_class")
    assert _instrument_tick(tmp_path, stock) == (0.01, "exchange_asset_class")
    assert _instrument_tick(tmp_path, override) == (0.005, "registry")


def test_market_scan_and_daily_backtest_use_same_etf_tick_and_source(tmp_path):
    instrument = InstrumentFile("510051", "etf", str(tmp_path / "data/etfs/510051.csv"))
    _write_bars(tmp_path / instrument.path)

    scanned = load_daily_bars(instrument, root=tmp_path)
    replayed = load_research_bars(instrument, root=tmp_path)

    assert scanned.bars.attrs["quality"]["tick_size"] == replayed["tick_size"] == 0.001
    assert scanned.bars.attrs["quality"]["tick_size_source"] == replayed["tick_size_source"] == "exchange_asset_class"

    registry = tmp_path / "caisen-system/config/instruments.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"instruments": {"etf:510051": {"tick_size": 0.01}}}))
    overridden = load_daily_bars(instrument, root=tmp_path)
    assert overridden.bars.attrs["quality"]["tick_size"] == 0.01
    assert overridden.bars.attrs["quality"]["tick_size_source"] == "registry"
    assert overridden.data_version != scanned.data_version


def test_default_tick_and_explicit_override_produce_different_recognition_cache_keys(tmp_path):
    instrument = InstrumentFile("510051", "etf", str(tmp_path / "data/etfs/510051.csv"))
    default = {"data_revision": "mtime_ns:1,size:2", "normalized_rows": 1,
               "used_last_date": "2026-01-01", "tick_size": 0.001,
               "tick_size_source": "exchange_asset_class"}
    override = dict(default, tick_size=0.01, tick_size_source="registry")

    default_key, _ = _signal_cache_identity(instrument, default, "2026-01-01", None, "boundary_first")
    override_key, _ = _signal_cache_identity(instrument, override, "2026-01-01", None, "boundary_first")

    assert default_key != override_key


@pytest.mark.parametrize("tick_size", [0, -0.001, float("nan"), float("inf"), -float("inf")])
def test_registry_rejects_non_positive_or_non_finite_tick_size(tmp_path, tick_size):
    instrument = InstrumentFile("510050", "etf", str(tmp_path / "510050.csv"))

    with pytest.raises(ValueError, match="positive finite"):
        resolve_instrument_tick(tmp_path, instrument, {"tick_size": tick_size})
