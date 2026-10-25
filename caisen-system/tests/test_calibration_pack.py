from types import SimpleNamespace

import pandas as pd

from caisen.calibration_pack import build_blind_cases, render_review_html, stratified_quotas
from caisen.market import InstrumentFile


def test_stratified_quotas_are_exact_even_and_reproducible():
    first = stratified_quotas(10, seed=13)
    second = stratified_quotas(10, seed=13)

    assert first == second
    assert sum(first.values()) == 10
    assert set(first) == {(kind, timeframe) for kind in ("stock", "etf") for timeframe in ("1d", "1w")}
    assert max(first.values()) - min(first.values()) <= 1


def test_review_html_shows_price_volume_but_hides_machine_labels_and_exports_answers():
    html = render_review_html([
        {
            "case_id": "blind-001",
            "code": "000001",
            "name": "样本",
            "kind": "stock",
            "timeframe": "1d",
            "asof": "2026-10-23",
            "bars": [
                {"date": "2026-10-22", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
                {"date": "2026-10-23", "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 120},
            ],
        }
    ], seed=7)

    assert "盲标校准" in html
    assert "000001" in html
    assert "收盘价" in html
    assert 'value="forming"' in html
    assert 'value="confirmed"' in html
    assert '["N", "没有任何书本形态"]' in html
    assert "下载标注" in html
    assert "machine_snapshot" not in html
    assert "pattern_type" not in html
    assert "blind-001" in html


def test_new_review_pack_only_exposes_active_ten_patterns_and_versions_answers():
    html = render_review_html([
        {"case_id": "blind-001", "code": "000001", "kind": "stock", "timeframe": "1d",
         "asof": "2026-10-23", "bars": []}
    ], seed=42)

    assert '"calibration_round":"round_03_v15"' in html
    assert '"calibration_pack_version":"caisen-calibration-v2"' in html
    assert '"engine_version":"caisen-v15"' in html
    assert '"source_rule_version":"caisen-rules-v6"' in html
    assert '"active_pattern_codes":["W","S","BF+","IHS","TB","M","FB","BF-","HS","TT"]' in html
    assert "W底破底翻" not in html
    assert "假突破（头肩顶）" not in html
    assert '"WS"' not in html
    assert '"FBHS"' not in html


def test_review_html_escapes_untrusted_case_metadata():
    html = render_review_html([
        {
            "case_id": "blind-002",
            "code": "000002",
            "name": "</script><script>alert(1)</script>",
            "kind": "etf",
            "timeframe": "1w",
            "asof": "2026-10-23",
            "bars": [],
        }
    ], seed=9)

    assert r"\u003c/script\u003e\u003cscript\u003e" in html
    assert "</script><script>alert(1)" not in html


def test_build_blind_cases_uses_all_four_strata_without_reusing_instruments(tmp_path, monkeypatch):
    instruments = [
        InstrumentFile("000001", "stock", str(tmp_path / "000001.csv")),
        InstrumentFile("000002", "stock", str(tmp_path / "000002.csv")),
        InstrumentFile("510300", "etf", str(tmp_path / "510300.csv")),
        InstrumentFile("510500", "etf", str(tmp_path / "510500.csv")),
    ]
    daily_bars = pd.DataFrame({
        "date": pd.bdate_range("2025-01-03", periods=240),
        "open": [10.0] * 240, "high": [11.0] * 240, "low": [9.0] * 240,
        "close": [10.5] * 240, "volume": [100.0] * 240,
    })
    weekly_bars = pd.DataFrame({
        "date": pd.date_range("2025-01-03", periods=60, freq="W-FRI"),
        "open": [10.0] * 60, "high": [11.0] * 60, "low": [9.0] * 60,
        "close": [10.5] * 60, "volume": [100.0] * 60,
    })
    monkeypatch.setattr("caisen.calibration_pack.discover_instruments", lambda root: instruments)
    monkeypatch.setattr("caisen.calibration_pack.load_daily_bars", lambda instrument, root: SimpleNamespace(
        quality_status="usable", bars=daily_bars.copy(), instrument=instrument
    ))
    monkeypatch.setattr("caisen.calibration_pack.aggregate_weekly", lambda result, calendar: SimpleNamespace(bars=weekly_bars.copy()))
    monkeypatch.setattr("caisen.calibration_pack.default_china_calendar", lambda: object())

    cases = build_blind_cases(tmp_path, count=4, seed=13)

    assert len(cases) == 4
    assert len({case["code"] for case in cases}) == 4
    assert {(case["kind"], case["timeframe"]) for case in cases} == {
        (kind, timeframe) for kind in ("stock", "etf") for timeframe in ("1d", "1w")
    }
