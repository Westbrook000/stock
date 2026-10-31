from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from caisen import api
from caisen.market import InstrumentFile, MarketDataResult
from caisen.schemas import Pattern
from caisen.main_bottoms import link_main_bottom_children
from caisen.api import _decision_patterns_by_timeframe
from caisen.patterns import current_pattern_snapshot


def _bars(dates):
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates), "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.0, "volume": 1000.0, "name": "样本标的",
    })
    frame.attrs["quality"] = {"tick_size": 0.01, "lot_size": 100, "category": "equity"}
    frame.attrs["provenance"] = {"data_version": "fixture-v1"}
    return frame


def test_scan_uses_full_prefixes_for_main_path_without_legacy_fallback(monkeypatch):
    instrument = InstrumentFile("000001", "stock", "one.csv")
    daily = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    weekly = _bars(pd.date_range("2014-01-03", "2026-10-23", freq="W-FRI"))
    main = Pattern(
        pattern_id="main", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="confirmed", formed_at="2023-01-01", confirmed_at=300,
        confirmation_date="2025-01-01", invalidation=8.0, targets=[14.0], volume_verified=False,
        evidence=[{"code":"main_structure","value":{"role":"main","selected":True,
                   "signal_eligible":True,"scale":"5y","first_trough_date":"2023-01-01",
                   "frozen_at":"2023-01-01","structure_end_date":"2026-10-28"}},
                  {"code":"main_selection","value":{"selected":True,"decided_at":"2026-10-28"}}],
    )
    ended_main = Pattern(
        pattern_id="main-ended", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="invalidated", formed_at="2021-01-01", confirmed_at=None,
        evidence=[{"code":"main_structure","value":{"role":"main","signal_eligible":False}},
                  {"code":"main_selection","value":{"selected":False,"decided_at":"2026-10-28"}}],
    )
    child = Pattern(
        pattern_id="child", code="000001", kind="stock", timeframe="1d", pattern_type="bull_flag",
        direction="bullish", state="confirmed", formed_at="2025-01-01", confirmed_at=480,
        confirmation_date="2026-09-01", invalidation=9.0, targets=[13.0], volume_verified=True,
        last_observed_date="2026-10-28",
        key_points={"pivot_1_index":480,"pivot_1":9.8,"pivot_2_index":495,"pivot_2":10.2},
    )
    seen = {"main": {}, "main_background": {}}
    monkeypatch.setattr(api, "discover_instruments", lambda root: [instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda item, root: MarketDataResult(item, bars=daily))
    def aggregate(result, **kwargs):
        seen["aggregate"] = result.bars.date.min(), result.bars.date.max()
        return MarketDataResult(result.instrument, bars=weekly, quality_status="usable")
    monkeypatch.setattr(api, "aggregate_weekly", aggregate)
    def detect_main(frame, code, kind, timeframe, **kwargs):
        seen["main"][timeframe] = frame.date.min(), frame.date.max(), len(frame)
        background=kwargs.get("background_bars")
        seen["main_background"][timeframe] = (background.date.min(), background.date.max(), len(background))
        return [main, ended_main] if timeframe == "1d" else []
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("scan must not call the legacy recognizer")))
    monkeypatch.setattr(api, "detect_main_bottoms", detect_main)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("main-bottom-fixture", {
        "timeframes": ["1d", "1w"], "date_from": "2014-01-01", "date_to": "2026-10-28",
    }, executor_factory=ThreadPoolExecutor)

    result = api._jobs["main-bottom-fixture"]
    assert seen["aggregate"] == (pd.Timestamp("2014-01-01"), pd.Timestamp("2026-10-28"))
    assert seen["main"]["1d"][0] == pd.Timestamp("2014-01-01")
    assert seen["main"]["1d"][1] == pd.Timestamp("2026-10-28")
    assert seen["main"]["1w"][0] == pd.Timestamp("2014-01-03")
    assert seen["main"]["1w"][1] == pd.Timestamp("2026-10-23")
    assert seen["main_background"]["1d"] == (daily.date.min(),daily.date.max(),len(daily))
    assert seen["main_background"]["1w"] == seen["main_background"]["1d"]
    row = result["results"][0]
    assert [event["pattern_id"] for event in row["main_structure_events"]] == ["main", "main-ended"]
    assert [event["pattern_id"] for event in row["selected_main_structures"]] == ["main"]
    assert row["main_children"] == []
    assert [event["pattern_id"] for event in row["ended_patterns"]] == ["main-ended"]
    assert row["pattern_migration"]["families"]["bull_flag"]["status"] == "unsupported"


def test_main_bottom_chart_offset_uses_calendar_cutoff():
    bars = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    offset = api._main_bottom_chart_offset(bars, "1d")
    assert bars.date.iloc[offset] >= pd.Timestamp("2021-10-28")
    assert bars.date.iloc[offset - 1] < pd.Timestamp("2021-10-28")


def test_new_scan_task_uses_v17_without_rewriting_v16_baseline(monkeypatch):
    from caisen import RULE_VERSION
    from caisen.v16_baseline import RULE_VERSION as V16_BASELINE_RULE_VERSION
    from caisen.store import get_scan_run

    class DeferredThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

    monkeypatch.setattr(api, "discover_instruments", lambda root: [])
    monkeypatch.setattr(api.threading, "Thread", DeferredThread)
    response = api.start_scan({"timeframes": ["1d"], "kinds": ["stock"]})
    saved = get_scan_run(response["job_id"])
    assert RULE_VERSION == "caisen-v17"
    assert saved["rule_version"] == "caisen-v17"
    assert api._jobs[response["job_id"]]["rule_version"] == "caisen-v17"
    assert V16_BASELINE_RULE_VERSION == "caisen-v16"


def test_analysis_maps_main_indices_to_full_daily_chart_but_keeps_legacy_offset(monkeypatch):
    instrument = InstrumentFile("000001", "stock", "one.csv")
    daily = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    weekly = _bars(pd.date_range("2010-01-08", "2026-10-23", freq="W-FRI"))
    main = Pattern(
        pattern_id="main-chart", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="forming", formed_at="2022-01-01", confirmed_at=None,
        key_points={"pivot_1_index":2,"pivot_1_date":"2021-10-29","pivot_1":8.0,
                    "pivot_2_index":20,"pivot_2_date":"2021-11-24","pivot_2":12.0,
                    "pivot_3_index":40,"pivot_3_date":"2021-12-22","pivot_3":8.5},
        evidence=[{"code":"main_structure","value":{"role":"main","scale":"5y",
                   "signal_eligible":False,"support_lower":8.0,"support_upper":8.5,"neckline":{"slope":0,"intercept":12}}},
                  {"code":"main_selection","value":{"selected":True,"decided_at":"2026-10-28"}},
                  {"code":"neckline_tests","value":[{"group_id":"touch-a","pivot_date":"2021-11-24",
                   "known_date":"2021-11-26","level":12.0,"valid":True}]}],
    )
    monkeypatch.setattr(api, "_market_data", lambda code, kind: MarketDataResult(instrument, bars=daily, quality_status="usable"))
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(instrument, bars=weekly, quality_status="usable"))
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    seen_background={}
    def detect_main(frame, code, kind, timeframe, **kwargs):
        assert (frame is daily) if timeframe == "1d" else (frame is weekly)
        background=kwargs.get("background_bars")
        seen_background[timeframe]=(background.date.min(),background.date.max(),len(background))
        return [main] if timeframe == "1d" else []
    monkeypatch.setattr(api, "detect_main_bottoms", detect_main)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_pattern", lambda pattern: None)

    result = api.analysis("000001", "stock", "1d")

    assert seen_background == {"1d":(daily.date.min(),daily.date.max(),len(daily)),
                               "1w":(daily.date.min(),daily.date.max(),len(daily))}
    daily_offset = api._main_bottom_chart_offset(daily, "1d")
    assert result["patterns"][0]["chart_offset"] == daily_offset
    assert result["decision"]["patterns"][0]["chart_offset"] == daily_offset
    assert result["_chart_offset"] == len(daily) - 500
    assert result["_bars"][daily_offset]["date"] == str(daily.date.iloc[daily_offset].date())


def test_snapshot_keeps_all_main_events_of_one_existing_family():
    def make_event(pattern_id, formed_at, is_main):
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
            direction="bullish", state="forming", formed_at=formed_at, confirmed_at=None,
            evidence=[{"code":"main_structure","value":{"role":"main"}}] if is_main else [],
        )

    older_main = make_event("main-old", "2021-01-01", True)
    newer_main = make_event("main-new", "2025-01-01", True)
    older_legacy = make_event("legacy-old", "2021-01-01", False)
    newer_legacy = make_event("legacy-new", "2025-01-01", False)

    result = current_pattern_snapshot([older_main, newer_main, older_legacy, newer_legacy])

    assert {pattern.pattern_id for pattern in result} == {"main-old", "main-new", "legacy-new"}


def test_current_decision_uses_selected_main_not_historical_confirmations():
    def event(pattern_id, direction, state, selected, role="main"):
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
            direction=direction, state=state, formed_at="2020-01-01", confirmed_at=10,
            evidence=([{"code":"main_structure","value":{"role":role,"signal_eligible":True}},
                      {"code":"main_selection","value":{"selected":selected}}] if role else []),
        )
    historical_bear = event("old-main", "bearish", "confirmed", False)
    selected_main = event("selected-main", "bullish", "forming", True)
    legacy = event("legacy", "bearish", "confirmed", False, role=None)

    selected = _decision_patterns_by_timeframe({"1d":[historical_bear,selected_main,legacy]})

    assert [item.pattern_id for item in selected["1d"]] == ["selected-main"]
