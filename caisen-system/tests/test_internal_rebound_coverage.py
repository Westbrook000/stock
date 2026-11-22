from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caisen.main_bottoms import (detect_main_bottoms, _internal_rebound_coverage,
                                 _recognition_experiment_cache_key)
from caisen.research_config import RecognitionExperimentConfig
from caisen.research_config import recognition_experiment


SNAPSHOT = Path("/tmp/caisen-structure-boundary-snapshot-20261112-v2/data/stocks")


def _matching_geometry_events(code, cutoff, first_trough, latest_anchor, strict, s1=False):
    source = SNAPSHOT / f"{code}.csv"
    if not source.exists():
        pytest.skip("frozen 2026-11-12 OHLC snapshot is unavailable")
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= cutoff].reset_index(drop=True)
    config = (RecognitionExperimentConfig(
        "s1-anchor-span" if s1 else "g1-internal-rebound",
        require_internal_rebound_coverage=True,
        require_minimum_anchor_span=s1) if strict else None)
    with recognition_experiment(config):
        found = detect_main_bottoms(bars, code=code, kind="stock", timeframe="1d")
    matching = []
    for pattern in found:
        evidence = {item["code"]: item["value"] for item in pattern.evidence}
        structure = evidence.get("main_structure", {})
        if (structure.get("first_trough_date") == first_trough
                and structure.get("latest_anchor_date") == latest_anchor):
            matching.append((pattern, evidence))
    return matching


def test_s1_integrated_gate_rejects_r02_short_w_anchors_after_long_wait():
    # R02's two actual W anchors are only 40 days apart. The later observation
    # date does not lengthen the structure's anchor span.
    g1 = _matching_geometry_events("000008", "2026-03-30", "2024-10-18",
                                   "2024-11-27", strict=True)
    s1 = _matching_geometry_events("000008", "2026-03-30", "2024-10-18",
                                   "2024-11-27", strict=True, s1=True)

    assert g1, "G1 should retain the known forming W geometry for R02"
    assert s1 == [], "S1 must reject this exact short-anchor geometry"


def _synthetic_w_bars():
    dates = pd.bdate_range("2021-01-04", periods=1300)
    anchors = [(0, 3.0), (80, 3.30), (160, 2.8), (240, 3.33),
               (320, 2.9), (390, 2.22), (470, 3.30), (570, 2.25),
               (660, 3.28), (760, 3.0), (820, 3.34), (930, 3.0),
               (1040, 3.5), (1100, 3.34), (1180, 4.4), (1299, 4.6)]
    x = np.arange(len(dates))
    close = np.interp(x, [p[0] for p in anchors], [p[1] for p in anchors])
    close += 0.012 * np.sin(x * 0.31) + 0.006 * np.sin(x * 0.071)
    bars = pd.DataFrame({"date": dates, "open": close, "high": close + 0.025,
                         "low": close - 0.025, "close": close,
                         "volume": 1_000_000 + 100_000 * np.sin(x * 0.17)})
    bars.loc[[390, 570], "low"] = [2.22, 2.25]
    bars.loc[[470, 820, 1100], "high"] = [3.30, 3.34, 3.34]
    return bars


def test_ihs_rebound_before_pressure_effective_start_is_still_covered():
    highs = np.array([14.0, 24.69, 15.0, 14.0, 14.2, 14.0, 13.9, 14.1, 14.0])
    atr = np.full(len(highs), 1.0)
    line = {"slope": 0.0, "intercept": 14.532794}

    # The left-shoulder to head interval starts before the line's effective
    # pressure segment. Geometry coverage must still inspect this rebound.
    report = _internal_rebound_coverage(highs, atr, line, [(0, 4), (4, 8)], 0.01)

    assert report["passed"] is False
    assert report["segments"][0]["worst_index"] == 1
    assert report["segments"][0]["maximum_excess"] > 9.6


def test_ihs_internal_rebound_outside_contemporaneous_tolerance_fails():
    highs = np.array([4.7, 5.53, 4.8, 4.9, 4.7])
    atr = np.full(len(highs), 0.2)
    line = {"slope": 0.0, "intercept": 5.012857}

    report = _internal_rebound_coverage(highs, atr, line, [(0, 2)], 0.01)

    assert report["passed"] is False
    assert report["segments"][0]["worst_index"] == 1


def test_internal_rebound_inside_tolerance_passes():
    highs = np.array([10.0, 10.05, 9.9])
    atr = np.full(len(highs), 0.1)
    line = {"slope": 0.0, "intercept": 10.0}

    report = _internal_rebound_coverage(highs, atr, line, [(0, 2)], 0.01)

    assert report["passed"] is True
    assert report["segments"][0]["maximum_excess"] <= 1e-12


def test_normal_rebound_below_frozen_line_passes():
    highs = np.array([9.0, 9.2, 9.35, 9.5, 9.6])
    atr = np.full(len(highs), 0.4)
    line = {"slope": 0.1, "intercept": 9.0}

    report = _internal_rebound_coverage(highs, atr, line, [(0, 2), (2, 4)], 0.01)

    assert report["passed"] is True


def test_experiment_switch_defaults_off_and_is_part_of_cache_identity():
    baseline = RecognitionExperimentConfig("rebound-coverage-v1")
    strict = RecognitionExperimentConfig(
        "rebound-coverage-v1", require_internal_rebound_coverage=True)

    assert baseline.require_internal_rebound_coverage is False
    assert strict.require_internal_rebound_coverage is True
    assert baseline.cache_key() != strict.cache_key()
    with recognition_experiment(baseline):
        baseline_key = _recognition_experiment_cache_key()
    with recognition_experiment(strict):
        strict_key = _recognition_experiment_cache_key()
    assert baseline_key != strict_key
    from caisen.daily_research import _recognition_experiment_metadata
    assert _recognition_experiment_metadata(strict)["require_internal_rebound_coverage"] is True
    from caisen.daily_research import _signal_cache_identity
    from caisen.market import InstrumentFile
    instrument = InstrumentFile("603009", "stock", "/snapshot/603009.csv")
    provenance = {"data_revision": "frozen", "normalized_rows": 310,
                  "used_last_date": "2015-11-12", "tick_size": 0.01}
    baseline_cache = _signal_cache_identity(instrument, provenance, "2015-11-12", None,
                                            "boundary_first", baseline)[0]
    strict_cache = _signal_cache_identity(instrument, provenance, "2015-11-12", None,
                                          "boundary_first", strict)[0]
    assert baseline_cache != strict_cache


def test_recognition_metadata_decodes_none_and_explicit_admission_types():
    from caisen.daily_research import _recognition_experiment_metadata

    unrestricted = RecognitionExperimentConfig(
        "g1-v1", minimum_neckline_tests=2, require_internal_rebound_coverage=True)
    restricted = RecognitionExperimentConfig(
        "g1-v1", minimum_neckline_tests=2, admission_types=frozenset({"w_bottom", "triangle_bottom"}),
        require_internal_rebound_coverage=True)

    assert "admission_types" not in _recognition_experiment_metadata(unrestricted)
    assert _recognition_experiment_metadata(restricted)["admission_types"] == ["triangle_bottom", "w_bottom"]


def test_recognition_metadata_decodes_current_and_legacy_cache_tuples():
    from caisen.daily_research import _recognition_experiment_metadata

    current = ("g1-v1", 2, "test_count_first", None, True)
    legacy = ("g1-v1", 2, ("w_bottom", "triangle_bottom"))

    current_metadata = _recognition_experiment_metadata(current)
    legacy_metadata = _recognition_experiment_metadata(legacy)
    assert "admission_types" not in current_metadata
    assert current_metadata["require_internal_rebound_coverage"] is True
    assert legacy_metadata["admission_types"] == ["triangle_bottom", "w_bottom"]


def test_minimum_anchor_span_experiment_is_opt_in_and_cache_keyed():
    baseline = RecognitionExperimentConfig("g1-v1", require_internal_rebound_coverage=True)
    strict = RecognitionExperimentConfig("s1-v1", require_internal_rebound_coverage=True,
                                         require_minimum_anchor_span=True)

    assert baseline.require_minimum_anchor_span is False
    assert strict.require_minimum_anchor_span is True
    assert baseline.cache_key() != strict.cache_key()


def test_s1_anchor_span_metadata_decodes_current_cache_tuple_and_keeps_old_tuple():
    from caisen.daily_research import _recognition_experiment_metadata

    current = ("s1-v1", None, "test_count_first", None, True, True)
    legacy_g1 = ("g1-v1", None, "test_count_first", None, True)
    old_tuple = ("legacy-v1", 2, ("w_bottom",))

    current_meta = _recognition_experiment_metadata(current)
    g1_meta = _recognition_experiment_metadata(legacy_g1)
    old_meta = _recognition_experiment_metadata(old_tuple)
    assert current_meta["require_minimum_anchor_span"] is True
    assert current_meta["require_internal_rebound_coverage"] is True
    assert g1_meta["require_minimum_anchor_span"] is False
    assert old_meta["admission_types"] == ["w_bottom"]


def test_minimum_anchor_span_uses_structure_anchors_and_calendar_boundaries_only():
    from caisen.main_bottoms import _minimum_anchor_span_evidence
    from caisen.schemas import Pivot

    def pivot(index, date, kind="low"):
        return Pivot(index, date, 10.0, kind, index + 1, date, "fixture")

    short_w = [pivot(0, "2024-10-18"), pivot(12, "2024-10-30", "high"),
               pivot(40, "2024-11-27")]
    rejected = _minimum_anchor_span_evidence("w_bottom", short_w,
        pd.date_range("2024-10-18", "2024-11-27", freq="D"), "1d")
    assert rejected["passed"] is False
    assert rejected["first_anchor_date"] == "2024-10-18"
    assert rejected["last_anchor_date"] == "2024-11-27"
    assert rejected["minimum_anchor_date"] == "2025-04-18"

    at_daily_boundary = [pivot(0, "2024-10-18"), pivot(182, "2025-04-18")]
    before_daily_boundary = [pivot(0, "2024-10-18"), pivot(181, "2025-04-17")]
    assert _minimum_anchor_span_evidence("w_bottom", at_daily_boundary,
        pd.date_range("2024-10-18", "2025-04-18", freq="D"), "1d")["passed"] is True
    assert _minimum_anchor_span_evidence("w_bottom", before_daily_boundary,
        pd.date_range("2024-10-18", "2025-04-17", freq="D"), "1d")["passed"] is False

    weekly = [pivot(0, "2024-01-31"), pivot(366, "2025-01-31")]
    assert _minimum_anchor_span_evidence("inverse_head_shoulders", weekly,
        pd.date_range("2024-01-31", "2025-01-31", freq="D"), "1w")["passed"] is True

    triangle = [pivot(0, "2024-10-18"), pivot(12, "2024-10-30", "high"),
                pivot(20, "2024-11-07"), pivot(30, "2024-11-17", "high"),
                pivot(40, "2024-11-27")]
    triangle_span = _minimum_anchor_span_evidence("triangle_bottom", triangle,
        pd.date_range("2024-10-18", "2024-11-27", freq="D"), "1d")
    assert triangle_span["passed"] is False
    assert triangle_span["anchor_role"] == "first_to_last_required_triangle_anchor"


def test_s1_exempts_flags_accepts_real_long_structures_and_ignores_future_observations():
    from caisen.main_bottoms import _minimum_anchor_span_evidence
    from caisen.schemas import Pivot

    first = Pivot(0, "2024-01-01", 10.0, "low", 1, "2024-01-02", "fixture")
    last = Pivot(182, "2024-07-01", 10.0, "low", 183, "2024-07-02", "fixture")
    exact_dates = pd.date_range("2024-01-01", "2024-07-01", freq="D")
    long_w = _minimum_anchor_span_evidence("w_bottom", [first, last], exact_dates, "1d")
    assert long_w["passed"] is True

    flag = _minimum_anchor_span_evidence("bull_flag", [first, last], exact_dates, "1d")
    assert flag["applies"] is False
    assert flag["passed"] is True

    short_first = Pivot(0, "2024-10-18", 10.0, "low", 1, "2024-10-19", "fixture")
    short_last = Pivot(40, "2024-11-27", 10.0, "low", 41, "2024-11-28", "fixture")
    prefix_dates = pd.date_range("2024-10-18", "2024-11-27", freq="D")
    future_dates = prefix_dates.append(pd.date_range("2024-11-28", "2025-10-18", freq="D"))
    prefix = _minimum_anchor_span_evidence("w_bottom", [short_first, short_last], prefix_dates, "1d")
    extended = _minimum_anchor_span_evidence("w_bottom", [short_first, short_last], future_dates, "1d")
    assert prefix == extended
    assert extended["passed"] is False


def test_603009_early_left_rebound_candidate_is_rejected_only_in_strict_experiment():
    baseline = _matching_geometry_events("603009", "2015-11-12", "2015-04-16",
                                         "2015-10-30", strict=False)
    strict = _matching_geometry_events("603009", "2015-11-12", "2015-04-16",
                                       "2015-10-30", strict=True)

    assert baseline
    assert baseline[0][1]["neckline_selection"]["selected_candidate_id"] == \
        "neckline-candidate-227-241-295"
    assert baseline[0][1]["main_structure"]["neckline"] == {
        "slope": -0.008382352941176474, "intercept": 16.24279411764706}
    assert strict == []


def test_000955_normal_rebound_structure_is_kept_with_strict_experiment():
    baseline = _matching_geometry_events("000955", "2004-04-07", "2003-09-16",
                                         "2004-03-09", strict=False)
    strict = _matching_geometry_events("000955", "2004-04-07", "2003-09-16",
                                       "2004-03-09", strict=True)

    assert baseline and strict
    assert baseline[0][1]["neckline_selection"]["selected_candidate_id"] == \
        strict[0][1]["neckline_selection"]["selected_candidate_id"]
    assert "internal_rebound_coverage" not in baseline[0][1]["neckline_selection"]
    assert strict[0][1]["neckline_selection"]["internal_rebound_coverage"]["passed"] is True


def test_w_bottom_strict_coverage_checks_all_bars_between_the_two_bottoms():
    bars = _synthetic_w_bars()
    with recognition_experiment(None):
        baseline = detect_main_bottoms(bars.copy(), code="fixture", kind="stock", timeframe="1d")
    config = RecognitionExperimentConfig("g1-w-internal-rebound",
                                         require_internal_rebound_coverage=True)
    with recognition_experiment(config):
        strict = detect_main_bottoms(bars.copy(), code="fixture", kind="stock", timeframe="1d")
    baseline_w = next(pattern for pattern in baseline if pattern.pattern_type == "w_bottom"
                      and next(item["value"] for item in pattern.evidence
                               if item["code"] == "main_structure")["role"] == "main")
    strict_w = next(pattern for pattern in strict if pattern.pattern_type == "w_bottom"
                    and next(item["value"] for item in pattern.evidence
                             if item["code"] == "main_structure")["role"] == "main")
    baseline_selection = next(item["value"] for item in baseline_w.evidence
                              if item["code"] == "neckline_selection")
    strict_selection = next(item["value"] for item in strict_w.evidence
                            if item["code"] == "neckline_selection")

    assert "internal_rebound_coverage" not in baseline_selection
    report = strict_selection["internal_rebound_coverage"]
    assert report["passed"] is True
    assert len(report["segments"]) == 1
    assert report["segments"][0]["start_index"] == strict_w.key_points["pivot_1_index"]
    assert report["segments"][0]["end_index"] == strict_w.key_points["pivot_2_index"]


def test_s1_keeps_a_genuinely_long_w_structure_and_records_anchor_dates():
    bars = _synthetic_w_bars()
    cutoff = bars[bars.date <= "2024-11-22"].copy()
    config = RecognitionExperimentConfig(
        "s1-g1-w-long", require_internal_rebound_coverage=True,
        require_minimum_anchor_span=True)

    with recognition_experiment(config):
        found = detect_main_bottoms(cutoff, code="fixture", kind="stock", timeframe="1d")

    selected = next(pattern for pattern in found if pattern.pattern_type == "w_bottom"
                    and next(item["value"] for item in pattern.evidence
                             if item["code"] == "main_structure")["role"] == "main")
    structure = next(item["value"] for item in selected.evidence if item["code"] == "main_structure")
    span = structure["minimum_anchor_span"]
    assert span["passed"] is True
    assert span["first_anchor_date"] == selected.key_points["first_trough_date"]
    assert span["last_anchor_date"] == selected.key_points["latest_anchor_date"]
    assert pd.Timestamp(span["minimum_anchor_date"]) <= pd.Timestamp(span["last_anchor_date"])
