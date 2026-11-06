import pandas as pd

from caisen.daily_incremental import DailyRecognitionSession
from caisen.research_config import RecognitionExperimentConfig, recognition_experiment
from caisen.schemas import Pivot


def test_prefix_frame_and_numeric_views_are_reused_within_one_causal_bar():
    session = DailyRecognitionSession("CACHE", "stock")
    session.append_bar({"date": "2026-01-05", "open": 10, "high": 10.2,
                        "low": 9.8, "close": 10.0, "volume": 1000})

    first = session._frame_for_current_prefix()
    session._prepare_neckline_frame(first)
    second = session._frame_for_current_prefix()

    assert first is second
    assert session.performance["prefix_frame_builds"] == 1
    assert first.attrs["neckline_low_values"].tolist() == [9.8]
    assert first.attrs["neckline_date_index"].tolist() == [pd.Timestamp("2026-01-05")]

    session.append_bar({"date": "2026-01-06", "open": 10, "high": 10.3,
                        "low": 9.9, "close": 10.1, "volume": 1100})
    extended = session._frame_for_current_prefix()

    assert extended is not first
    assert len(extended) == 2
    assert session.performance["prefix_frame_builds"] == 2


def test_daily_pressure_prefilter_honors_experiment_minimum_across_prior_and_after_touches():
    session = DailyRecognitionSession("MIN3", "stock")
    dates = pd.bdate_range("2022-01-03", periods=50)
    session.bars = [{"date": date} for date in dates]
    session.date_strings = [str(date.date()) for date in dates]
    session.atr_values = [0.5] * len(dates)
    first = Pivot(10, session.date_strings[10], 2.0, "low", 12,
                  session.date_strings[12], "3ATR|8%")
    middle = Pivot(20, session.date_strings[20], 3.0, "high", 22,
                   session.date_strings[22], "3ATR|8%")
    between = Pivot(15, session.date_strings[15], 2.0, "low", 17,
                    session.date_strings[17], "3ATR|8%")
    last = Pivot(40, session.date_strings[40], 2.1, "low", 42,
                 session.date_strings[42], "3ATR|8%")
    pressure_left = Pivot(5, session.date_strings[5], 3.0, "high", 7,
                          session.date_strings[7], "neckline_test_2ATR")
    pressure_right = Pivot(35, session.date_strings[35], 3.0, "high", 37,
                           session.date_strings[37], "neckline_test_2ATR")
    session.scale_pivots["3atr_8pct"] = [first, middle, between, last]
    session.pressure["pivots"] = [pressure_left, pressure_right]
    for pivot in (first, between, last):
        session._index_low_pivot(pivot)
    candidate = {"family": "w_bottom", "sequence": [first, last], "pattern": None}

    assert session._has_pressure_test_pair(candidate)
    config = RecognitionExperimentConfig(
        "neckline_min3_v1", minimum_neckline_tests=3,
        admission_types=frozenset({"w_bottom", "inverse_head_shoulders",
                                   "triangle_bottom", "bull_flag"}))
    with recognition_experiment(config):
        assert not session._has_pressure_test_pair(candidate)
