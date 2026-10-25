from pathlib import Path

import hashlib

import pandas as pd

from caisen.market import (
    InstrumentFile,
    MarketDataResult,
    aggregate_weekly,
    audit_market_inventory,
    discover_instruments,
    load_daily_bars,
)


def test_discovery_returns_stably_sorted_stock_and_etf_files(tmp_path):
    (tmp_path / "data" / "stocks").mkdir(parents=True)
    (tmp_path / "data" / "etfs").mkdir(parents=True)
    (tmp_path / "data" / "stocks" / "000002.csv").write_text("date,close\n", encoding="utf-8")
    (tmp_path / "data" / "stocks" / "000001.csv").write_text("date,close\n", encoding="utf-8")
    (tmp_path / "data" / "etfs" / "510300.csv").write_text("date,close\n", encoding="utf-8")
    (tmp_path / "data" / "stocks" / "ignored.txt").write_text("x", encoding="utf-8")

    instruments = discover_instruments(tmp_path)

    assert all(isinstance(item, InstrumentFile) for item in instruments)
    assert [(item.kind, item.code) for item in instruments] == [
        ("stock", "000001"), ("stock", "000002"), ("etf", "510300")
    ]
    assert all(Path(item.path).is_file() and item.status == "discovered" for item in instruments)


def test_discovery_returns_empty_for_missing_data_directories(tmp_path):
    assert discover_instruments(tmp_path) == []


def write_basis(root, stock="qfq", etf="raw"):
    directory = root / "data" / "market_data_current"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".market-data-basis.json").write_text(
        '{"adjustment_by_kind":{"stock":"%s","etf":"%s"}}' % (stock, etf),
        encoding="utf-8",
    )


def write_stock_csv(root, rows):
    folder = root / "data" / "stocks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "000001.csv"
    path.write_text(
        "\ufeff日期,代码,名称,开盘,最高,最低,收盘,成交量\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_daily_bars_preserves_file_hash_adjustment_and_cutoff(tmp_path):
    write_basis(tmp_path)
    path = write_stock_csv(tmp_path, [
        "2026-01-05,000001,样本,10,11,9,10.5,100",
        "2026-01-06,000001,样本,10.5,12,10,11.5,120",
    ])
    instrument = discover_instruments(tmp_path)[0]

    result = load_daily_bars(instrument, root=tmp_path)

    assert result.instrument == instrument
    assert result.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.adjustment_basis == "qfq"
    assert result.first_date == "2026-01-05"
    assert result.last_date == "2026-01-06"
    assert result.source == "local_csv"
    assert result.bars.close.tolist() == [10.5, 11.5]
    assert result.data_version
    assert result.quality_status == "usable_with_warnings"
    assert not any("code mismatch" in warning for warning in result.warnings)


def test_load_daily_bars_drops_and_reports_bad_rows_and_duplicates(tmp_path):
    write_basis(tmp_path)
    write_stock_csv(tmp_path, [
        "2026-01-05,000001,样本,10,11,9,10,100",
        "not-a-date,000001,样本,10,11,9,10,100",
        "2026-01-06,000001,样本,10,11,9,10,-1",
        "2026-01-07,000001,样本,10,8,9,10,100",
        "2026-01-05,000001,样本,10,12,9,11,100",
        "2026-01-08,999999,别的代码,10,11,9,10,100",
    ])
    instrument = discover_instruments(tmp_path)[0]

    result = load_daily_bars(instrument, root=tmp_path)

    assert result.rejected_rows == 3
    assert result.bars.date.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08")]
    assert result.bars.close.iloc[0] == 11
    assert any("duplicate" in warning for warning in result.warnings)
    assert any("code mismatch" in warning for warning in result.warnings)
    assert result.quality_status == "usable_with_warnings"


def test_as_of_version_ignores_future_rows_but_raw_file_hash_does_not(tmp_path):
    write_basis(tmp_path)
    path = write_stock_csv(tmp_path, [
        "2026-01-05,000001,样本,10,11,9,10,100",
        "2026-01-06,000001,样本,10,12,9,11,100",
    ])
    instrument = discover_instruments(tmp_path)[0]
    first = load_daily_bars(instrument, root=tmp_path, as_of="2026-01-05")
    path.write_text(path.read_text(encoding="utf-8-sig") + "2026-01-07,000001,样本,11,13,10,12,150\n", encoding="utf-8-sig")
    second = load_daily_bars(instrument, root=tmp_path, as_of="2026-01-05")

    assert first.file_sha256 != second.file_sha256
    assert first.data_version == second.data_version
    assert second.last_date == "2026-01-05"
    assert second.bars.date.max() == pd.Timestamp("2026-01-05")


class FixtureCalendar:
    version = "fixture-calendar-v1"

    def __init__(self, last_sessions):
        self.last_sessions = last_sessions

    def last_session_of_week(self, week_start):
        return self.last_sessions.get(str(week_start))


def make_bars(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": [10.0] * len(dates), "high": [11.0] * len(dates),
        "low": [9.0] * len(dates), "close": [10.0] * len(dates),
        "volume": [100.0] * len(dates), "segment_id": [0] * len(dates),
    })


def test_weekly_short_week_requires_calendar_and_exposes_preview():
    source = make_bars(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])

    unknown = aggregate_weekly(source, asof="2026-01-09")
    known = aggregate_weekly(
        source,
        asof="2026-01-09",
        calendar=FixtureCalendar({"2026-01-05": "2026-01-08"}),
    )

    assert unknown.empty
    assert len(unknown.attrs["preview_bars"]) == 1
    assert len(known) == 1
    assert known.iloc[0].date == pd.Timestamp("2026-01-08")


def test_weekly_without_verified_calendar_stays_preview_even_with_friday_bar():
    source = make_bars(pd.bdate_range("2026-01-05", periods=5))
    weekly = aggregate_weekly(source, asof="2026-01-09")

    assert weekly.empty
    assert len(weekly.attrs["preview_bars"]) == 1
    assert weekly.attrs["quality"]["weekly_calendar_verified"] is False


def test_weekly_as_of_midweek_does_not_confirm_friday_week():
    source = make_bars(["2026-01-05", "2026-01-06", "2026-01-07"])

    weekly = aggregate_weekly(source, asof="2026-01-07")

    assert weekly.empty
    assert len(weekly.attrs["preview_bars"]) == 1
    assert weekly.attrs["quality"]["weekly_status"] == "incomplete_week_preview"


def test_weekly_result_carries_provisional_bars_and_calendar_version():
    instrument = InstrumentFile("000001", "stock", "/tmp/000001.csv")
    daily = MarketDataResult(
        instrument=instrument,
        bars=make_bars(["2026-01-05", "2026-01-06", "2026-01-07"]),
        data_version="daily-v1",
        quality_status="usable_with_warnings",
    )

    weekly = aggregate_weekly(daily, calendar=FixtureCalendar({"2026-01-05": "2026-01-09"}), asof="2026-01-07")

    assert isinstance(weekly, MarketDataResult)
    assert weekly.bars.empty
    assert len(weekly.provisional_bars) == 1
    assert weekly.calendar_version == "fixture-calendar-v1"
    assert weekly.quality_status == "insufficient"


def test_xshg_calendar_confirms_official_2026_short_weeks_and_stops_at_coverage_boundary():
    from datetime import date
    from caisen.market import ChinaXSHGCalendar

    calendar = ChinaXSHGCalendar()

    assert calendar.last_session_of_week(date(2026, 9, 21)) == pd.Timestamp("2026-09-24")
    assert calendar.last_session_of_week(date(2026, 10, 5)) == pd.Timestamp("2026-10-09")
    assert calendar.last_session_of_week(date(2027, 1, 4)) is None


def test_official_calendar_makes_holiday_shortened_week_formal_only_after_last_session():
    from caisen.market import ChinaXSHGCalendar

    source = make_bars(["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"])
    calendar = ChinaXSHGCalendar()

    before_close = aggregate_weekly(source, asof="2026-09-23", calendar=calendar)
    after_close = aggregate_weekly(source, asof="2026-09-24", calendar=calendar)

    assert before_close.empty
    assert len(before_close.attrs["preview_bars"]) == 1
    assert len(after_close) == 1
    assert after_close.iloc[0].date == pd.Timestamp("2026-09-24")


def test_pre2020_library_history_remains_weekly_preview():
    from caisen.market import ChinaXSHGCalendar

    source = make_bars(pd.bdate_range("2019-01-07", periods=5))
    weekly = aggregate_weekly(source, asof="2019-01-11", calendar=ChinaXSHGCalendar())

    assert weekly.empty
    assert len(weekly.attrs["preview_bars"]) == 1
    assert weekly.attrs["quality"]["weekly_calendar_verified"] is False


def test_sse_official_closure_data_drives_weekly_sessions_for_2020_to_2026():
    import json
    from datetime import date, timedelta
    from pathlib import Path
    from caisen.market import ChinaXSHGCalendar

    source_path = Path(__file__).parents[1] / "caisen" / "data" / "xshg-official-closures.json"
    official = json.loads(source_path.read_text(encoding="utf-8"))
    calendar = ChinaXSHGCalendar()

    for year in range(2020, 2027):
        closures = official["years"][str(year)]["closed_ranges"]
        closed = set()
        for first, last in closures:
            day = date.fromisoformat(first)
            end = date.fromisoformat(last)
            while day <= end:
                closed.add(day)
                day += timedelta(days=1)
        expected = {
            date(year, 1, 1) + timedelta(days=offset)
            for offset in range((date(year, 12, 31) - date(year, 1, 1)).days + 1)
            if (date(year, 1, 1) + timedelta(days=offset)).weekday() < 5
            and (date(year, 1, 1) + timedelta(days=offset)) not in closed
        }
        assert calendar.is_week_verified(date(year, 6, 1))
        monday = date(year, 1, 1)
        while monday.weekday() != 0:
            monday += timedelta(days=1)
        while monday.year == year and (monday + timedelta(days=4)).year == year:
            expected_week = [monday + timedelta(days=i) for i in range(5) if monday + timedelta(days=i) in expected]
            actual_last = calendar.last_session_of_week(monday)
            assert actual_last == (pd.Timestamp(expected_week[-1]) if expected_week else None)
            monday += timedelta(days=7)

    # Fixed official-calendar anchors guard against a source file that is
    # internally consistent but contains a copied or mis-entered closure.
    assert calendar.last_session_of_week(date(2020, 1, 27)) == pd.Timestamp("2020-01-31")
    assert calendar.last_session_of_week(date(2025, 1, 27)) == pd.Timestamp("2025-01-27")
    assert calendar.last_session_of_week(date(2025, 5, 5)) == pd.Timestamp("2025-05-09")
    assert calendar.last_session_of_week(date(2026, 9, 21)) == pd.Timestamp("2026-09-24")
    assert calendar.last_session_of_week(date(2027, 1, 4)) is None


def test_full_inventory_reports_discovered_readable_warning_insufficient_failed(tmp_path):
    stocks = tmp_path / "data" / "stocks"
    etfs = tmp_path / "data" / "etfs"
    stocks.mkdir(parents=True)
    etfs.mkdir(parents=True)
    valid = "日期,代码,名称,开盘,最高,最低,收盘,成交量\n2026-01-05,000001,A,10,11,9,10,100\n"
    (stocks / "000001.csv").write_text(valid, encoding="utf-8-sig")
    (stocks / "000002.csv").write_text("日期,代码,名称\n2026-01-05,000002,B\n", encoding="utf-8-sig")
    (etfs / "510001.csv").write_text("", encoding="utf-8-sig")

    report = audit_market_inventory(tmp_path)

    assert report["discovered"] == 3
    assert report["readable"] == 1
    assert report["warning"] == 1
    assert report["insufficient"] == 0
    assert report["failed"] == 2
    assert report["stock_count"] == 2
    assert report["etf_count"] == 1
    assert len(report["files"]) == report["discovered"]
