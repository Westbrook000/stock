import unittest

import pandas as pd

from caisen.market import aggregate_weekly, load_local_history, normalize_bars


def bars(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": [10.0] * len(dates),
        "high": [11.0] * len(dates), "low": [9.0] * len(dates),
        "close": [10.0] * len(dates), "volume": [100.0] * len(dates),
    })


class NormalWeekCalendar:
    version = "test-normal-week-v1"

    def last_session_of_week(self, week_start):
        return pd.Timestamp(week_start) + pd.Timedelta(days=4)


class MarketTests(unittest.TestCase):
    def test_normalize_sorts_deduplicates_and_reports_legacy_provenance(self):
        source = bars(["2026-01-06", "2026-01-05", "2026-01-06"])
        source.loc[2, "close"] = 10.5
        result = normalize_bars(source, code="000001", kind="stock", source="test")
        self.assertEqual(result["date"].tolist(), list(pd.to_datetime(["2026-01-05", "2026-01-06"])))
        self.assertEqual(result.iloc[-1]["close"], 10.5)
        self.assertTrue(result.attrs["quality"]["volume_unit_verified"] is False)

    def test_weekly_ohlcv_uses_last_daily_close_and_drops_open_week(self):
        source = bars(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"])
        source.loc[4, "close"] = 12.0
        source.loc[4, "high"] = 13.0
        source.loc[5, "close"] = 14.0
        result = aggregate_weekly(source, asof="2026-01-12", calendar=NormalWeekCalendar())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["close"], 12.0)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2026-01-09"))

    def test_short_holiday_week_records_last_real_session_date(self):
        source = bars(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-12"])
        class HolidayCalendar:
            version = "test-holiday-calendar-v1"

            def last_session_of_week(self, week_start):
                return {"2026-01-05": "2026-01-08", "2026-01-12": "2026-01-16"}.get(str(week_start))

        result = aggregate_weekly(source, asof="2026-01-12", calendar=HolidayCalendar())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2026-01-08"))

    def test_local_loader_reads_existing_chinese_csv_schema(self):
        result = load_local_history("000001", "stock", root=".")
        self.assertIsNotNone(result)
        self.assertIn("volume", result.columns)
        self.assertEqual(result.attrs["provenance"]["source"], "legacy_csv")

    def test_invalid_ohlc_is_rejected(self):
        source = bars(["2026-01-05"])
        source.loc[0, "high"] = 8.0
        with self.assertRaises(ValueError):
            normalize_bars(source, code="000001", kind="stock")

    def test_large_unverified_price_gap_starts_a_new_data_segment(self):
        source = bars(pd.bdate_range("2026-01-05", periods=4))
        source.loc[2, ["open", "high", "low", "close"]] = [16.0, 16.5, 15.5, 16.0]
        source.loc[3, ["open", "high", "low", "close"]] = [16.0, 16.5, 15.5, 16.0]
        result = normalize_bars(source, code="000001", kind="stock")
        self.assertEqual(result.segment_id.tolist(), [0, 0, 1, 1])
        self.assertEqual(result.attrs["quality"]["data_breaks"], [2])

    def test_weekly_aggregation_keeps_data_break_segments_separate(self):
        source = bars(pd.bdate_range("2026-01-05", periods=5))
        source.loc[2:, ["open", "high", "low", "close"]] = [16.0, 16.5, 15.5, 16.0]
        normalized = normalize_bars(source, code="000001", kind="stock")
        weekly = aggregate_weekly(normalized, asof="2026-01-09", calendar=NormalWeekCalendar())
        self.assertEqual(len(weekly), 1)
        self.assertEqual(weekly.segment_id.tolist(), [1])
        self.assertEqual(weekly.volume.tolist(), [300.0])


if __name__ == "__main__":
    unittest.main()
