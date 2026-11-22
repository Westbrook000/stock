import pandas as pd
import pytest

from caisen import api
from caisen.decision import synthesize_decision
from caisen.daily_incremental import DailyRecognitionSession, recognize_daily_history
from caisen.schemas import Pattern, Pivot


def forming(family="w_bottom", timeframe="1d"):
    return Pattern(pattern_id="case", code="test", kind="stock", timeframe=timeframe,
                   pattern_type=family, direction="bullish", state="forming",
                   formed_at="2020-01-03", confirmed_at=None, neckline={"slope":0., "intercept":10.},
                   invalidation=1., key_points={"first_trough_index":0,
                       "pivot_1_index":0, "pivot_1_date":"2020-01-03", "pivot_1":1.},
                   evidence=[{"code":"main_structure","value":{"role":"main",
                       "eligibility_reason":"awaiting_fresh_neckline_breakout"}},
                       {"code":"neckline_selection","value":{"line_frozen_at":"2020-01-03",
                        "structure_known_at":"2020-01-03"}}])


@pytest.mark.parametrize("period,expected", [("1d","2020-01-03"),("1w","2015-01-03")])
def test_window_and_causal_prior_context(period, expected):
    bars = pd.DataFrame({"date":pd.to_datetime(["2013-02-01","2014-04-01","2015-01-03",
        "2019-07-30","2020-01-03","2025-01-03"]),
        "low":[1.,3.,2.,3.,3.,4.],"high":[99.,16.,6.,9.,12.,100.]})
    frame = api._recognition_window(bars, period)
    assert str(frame.date.iloc[0].date()) == expected
    context = frame.attrs["prior_price_context"]
    assert (context["low"], context["high"]) == (1,99)
    assert context["two_year_high"][expected] == (9 if period=="1d" else 99)


@pytest.mark.parametrize("close,expected", [(7,"WATCH"),(6.99,"WAIT"),
    (10,"WAIT"),(10.2,"WAIT"),(9.9,"WATCH")])
def test_watch_only_strictly_below_frozen_boundary_within_three_atr(close, expected):
    p = forming()
    bars = pd.DataFrame([{"date":"2025-01-03","close":close}])
    result = synthesize_decision("test","stock",{"1d":[p]}, {"1d":close}, {"1d":"usable"},
        bars_by_timeframe={"1d":bars},
        watch_by_timeframe={"1d":{"date":"2025-01-03","index":100,"previous_atr":1}})
    assert result["action"] == expected
    assert not result["order_submitted"]
    if expected == "WATCH":
        assert result["watch"][0]["distance_atr"] == pytest.approx(10-close)


@pytest.mark.parametrize("period,anniversary,day_after", [
    ("1d","2025-01-03","2025-01-06"),("1w","2030-01-03","2030-01-10")])
def test_window_rollover_ends_unconfirmed_pattern_once(period, anniversary, day_after):
    s = DailyRecognitionSession("test","stock",timeframe=period)
    p = forming(timeframe=period)
    point = Pivot(0,"2020-01-03",1.,"low",0,"2020-01-03","2ATR")
    s.active["case"] = {"sequence":[point],"pattern":p,"structure_known_at":0}
    for day in ["2020-01-03",anniversary]:
        s.bars.append({"date":day,"close":3.})
        s.date_strings.append(day)
    s._maintain_candidates()
    assert p.state == "forming"
    s.bars.append({"date":day_after,"close":3.})
    s.date_strings.append(day_after)
    s._maintain_candidates()
    s._maintain_candidates()
    assert p.state == "invalidated"
    assert p.confirmation_date is None
    [archived] = s.retired_patterns.values()
    lifecycle = next(e["value"] for e in archived.evidence if e["code"]=="joint_lifecycle")
    events = [e for e in lifecycle["events"] if e["kind"]=="invalidated"]
    assert [(e["date"],e["reason"]) for e in events] == [(day_after,"structure_window_ended")]


@pytest.mark.parametrize("i,close,reason", [
    (4, 11.1, "triangle_first_breakout_outside_window"),
    (5, 11.1, None),
    (7, 11.1, None),
    (8, 9.8, "triangle_breakout_window_ended"),
    (8, 11.1, "triangle_first_breakout_outside_window"),
])
def test_triangle_breakout_window(i, close, reason):
    session = DailyRecognitionSession("test","stock")
    session.atr_values = [1.] * 10
    assert session._triangle_window_reason(forming("triangle_bottom"),
        {"targets":{"triangle_apex_index":10}}, i, close) == reason


def test_out_of_window_flag_archived_once_even_without_usable_atr():
    s = DailyRecognitionSession("test","stock")
    day = "2025-01-03"
    p = forming("bull_flag")
    p.evidence.append({"code":"flag_lifecycle","value":{"state":"forming","known_at":"2020-01-03"}})
    seq = tuple(Pivot(i,"2020-01-03",float(i+1),kind,i,"2020-01-03","2ATR")
                for i,kind in enumerate(("low","high","low","high","low")))
    key = tuple(pivot.index for pivot in seq)
    p.pattern_id = "test:bull_flag:" + "-".join(f"{pivot.index}:{pivot.kind}" for pivot in seq)
    s.flag_formations[p.pattern_id] = p
    s.flag_candidates[key] = seq
    s.bars = [{"date":day}] * 506
    s.date_strings = [day] * 506
    s.atr_values = [float("nan")] * 506
    assert s._check_bull_flag() == []
    assert s._check_bull_flag() == []
    assert p.state == "invalidated"
    [invalidated] = p.evidence[-1]["value"]["events"][-1:]
    assert (invalidated["date"], invalidated["reason"]) == (day, "flag_search_window_exceeded")


def test_watch_does_not_use_mismatched_or_incomplete_week():
    p = forming(timeframe="1w")
    bars = pd.DataFrame([{"date":"2025-01-03","close":9.}])
    for week, context in [(bars.iloc[:0], {"date":"2025-01-03","index":100,"previous_atr":1.}),
                          (bars, {"date":"2025-01-07","index":101,"previous_atr":1.})]:
        result = synthesize_decision("test","stock",{"1w":[p]}, {"1w":9.}, {"1w":"usable"},
            bars_by_timeframe={"1w":week}, watch_by_timeframe={"1w":context})
        assert result["action"] == "WAIT"


def test_scan_history_filters_watch_without_relabeling_wait(monkeypatch, tmp_path):
    from caisen.store import create_scan_run, save_scan_result, query_scan_results
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "scans.sqlite"))
    create_scan_run("test", {}, total=2, rule_version="test", app_version="test")
    for index, action in enumerate(["WATCH","WAIT"]):
        save_scan_result("test", index, {"code":str(index), "kind":"stock", "patterns":[], "action":action})
    rows, total = query_scan_results("test", action="WATCH")
    assert total == 1 and rows[0]["action"] == "WATCH"


def test_buy_priority_and_watch_uses_the_frozen_instance(monkeypatch):
    from caisen import decision

    p = forming()
    p.evidence.append({"code":"joint_lifecycle","value":{"state_at":"forming",
        "paired_at":"2025-01-02", "boundary":{"slope":0., "intercept":9.5}}})
    buy = forming("bull_flag","1w")
    buy.pattern_id, buy.state, buy.confirmation_date = "buy", "confirmed", "2025-01-03"
    monkeypatch.setattr(decision, "build_trade_plan",
                        lambda *a, **kw: {"pattern_id":"buy","decision":"CONSIDER_BUY"})
    bars = pd.DataFrame([{"date":"2025-01-03","close":9.}])
    result = synthesize_decision("test","stock",{"1d":[p], "1w":[buy]},
        {"1d":9., "1w":9.}, {"1d":"usable", "1w":"usable"},
        bars_by_timeframe={"1d":bars, "1w":bars},
        watch_by_timeframe={"1d":{"date":"2025-01-03","index":100,"previous_atr":1.}})
    assert result["action"] == "CONSIDER_BUY"
    assert result["watch"][0]["distance_atr"] == 0.5
    watch = {**result, "action":"WATCH", "rank":{"action_priority":decision._ACTION_ORDER["WATCH"]}}
    waiting = {**result, "action":"WAIT", "rank":{"action_priority":decision._ACTION_ORDER["WAIT"]}}
    assert [row["action"] for row in decision.rank_scan_results([waiting, watch, result])] == [
        "CONSIDER_BUY", "WATCH", "WAIT"]


def test_scan_and_detail_report_same_current_watch(monkeypatch):
    from caisen.market import InstrumentFile, MarketDataResult
    instrument = InstrumentFile("test", "stock", "fixture")
    day = pd.Timestamp("2025-01-03")
    bars = pd.DataFrame([{"date":day,"open":9.,"high":9.5,"low":8.5,"close":9.,"volume":100.}])
    result = MarketDataResult(instrument, bars=bars, quality_status="usable")
    weekly = MarketDataResult(instrument, bars=bars.iloc[:0], quality_status="insufficient")
    monkeypatch.setattr(api, "load_daily_bars", lambda *a, **kw: result)
    monkeypatch.setattr(api, "_market_data", lambda *a, **kw: result)
    monkeypatch.setattr(api, "aggregate_weekly", lambda *a, **kw: weekly)
    monkeypatch.setattr(api, "recognize_daily_history", lambda *a, **kw: {
        "patterns":[forming()], "signals":[],
        "latest_bar":{"date":"2025-01-03","index":0,"previous_atr":1.}})
    monkeypatch.setattr(api, "save_pattern", lambda *a, **kw: None)
    scan, _ = api._analyze_scan_instrument(instrument, {"timeframes":["1d"]})
    detail = api.analysis("test")["decision"]
    assert scan["action"] == detail["action"] == "WATCH"
    assert scan["watch"] == detail["watch"]
    assert not scan["strategy_events"]
