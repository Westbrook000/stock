"""Pin raw-bar evidence for the known 159122 delayed-recross rule defect."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd

from caisen.daily_incremental import generate_daily_signals, recognize_daily_instances
from caisen.daily_research import load_research_bars
from caisen.main_bottoms import detect_main_bottoms


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_research_review_20261105.py"
AUDIT_FIXTURES = Path(__file__).parent / "fixtures" / "delayed_maturity_159122"


def test_159122_audit_reports_first_pre_maturity_cross_and_later_emitted_recross(
        tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("review_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    archive = tmp_path / "archive"
    account_path = archive / "accounts" / "etf-159122.json"
    account_path.parent.mkdir(parents=True)
    shutil.copyfile(AUDIT_FIXTURES / "etf-159122.json", account_path)

    snapshot_path = tmp_path / "data" / "159122.csv"
    snapshot_path.parent.mkdir(parents=True)
    shutil.copyfile(AUDIT_FIXTURES / "159122-through-2026-06-22.csv", snapshot_path)
    account = json.loads(account_path.read_text(encoding="utf-8"))
    stat = snapshot_path.stat()
    revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
    source = {
        "account_data_revision": account["provenance"]["data_revision"],
        "snapshot_path": str(snapshot_path),
        "snapshot_root": str(tmp_path),
        "snapshot_data_revision": revision,
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(module, "DELAYED_BREAKOUT_ARCHIVE", archive)
    monkeypatch.setattr(module, "DATA_SOURCE_MAP", {"etf:159122": source})
    monkeypatch.setattr(module, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_VERIFIED_SNAPSHOT_HASHES", set())

    audit = module._delayed_maturity_breakout_audit()

    assert audit["status"] == "verified"
    assert audit["structure_known_at"] == "2026-04-16"
    assert audit["maturity_date"] == "2026-06-16"
    assert audit["first_buffered_close_after_structure_known"]["date"] == "2026-05-06"
    assert audit["first_buffered_close_after_structure_known"]["required_close"] < \
        audit["first_buffered_close_after_structure_known"]["close"]
    assert audit["signal_date"] == "2026-06-22"
    assert audit["signal_is_later_recross"] is True
    assert "rule_defect" in audit
    assert "Retire the candidate permanently" in audit["required_isolated_correction"]


def _159122_bars_through_june_2026():
    stock_root = Path(__file__).parents[2]
    path = stock_root / "data" / "etfs" / "159122.csv"
    loaded = load_research_bars(SimpleNamespace(code="159122", kind="etf", path=str(path)),
                                root=stock_root)
    bars = loaded["bars"]
    return bars[bars.date <= "2026-06-22"].reset_index(drop=True)


def test_159122_v12_does_not_create_a_late_buy_for_rejected_rebound_boundary():
    bars = _159122_bars_through_june_2026()

    result = recognize_daily_instances(bars, "159122", "etf", recognizer="boundary_first")
    target_ids = {event["instance_id"] for event in result["instance_events"]
                  if event["pattern"].get("pattern_type") == "w_bottom"
                  and event["pattern"].get("formed_at") == "2025-11-05"}
    # The v12 rebound-to-neckline association rule rejects this old candidate
    # before it can become a tradable instance. The same-boundary early-cross
    # and no-reissue lifecycle remains covered by the cached-candidate session
    # regression in test_daily_incremental_guards.py.
    assert not target_ids
    assert not any(signal["signal_date"] == "2026-06-22"
                   and "w_bottom" in signal["pattern_types"]
                   for signal in result["signals"])

    assert not any(signal["signal_date"] == "2026-06-22" for signal in result["signals"])


def test_boundary_phase_identity_tracks_frozen_geometry_not_later_touch_extensions():
    from caisen.main_bottoms import _same_neckline_boundary_phase

    original = {
        "line": {"slope": 0.0, "intercept": 1.1045},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 8, "coverage_end_index": 86}
                  for index in (8, 58, 71, 86)],
        "consumed_index": 120, "phase_tolerance": 0.02,
    }
    extended = {
        "line": {"slope": 1e-7, "intercept": 1.10449},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 8, "coverage_end_index": 120}
                  for index in (8, 58, 71, 86, 120)],
        "consumed_index": 120, "phase_tolerance": 0.02,
    }
    distinct = {
        "line": {"slope": 0.0, "intercept": 1.145},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 100, "coverage_end_index": 140}
                  for index in (100, 120, 140)],
        "consumed_index": 140, "phase_tolerance": 0.02,
    }

    assert not _same_neckline_boundary_phase(original, extended, tick_size=0.01)
    assert not _same_neckline_boundary_phase(original, distinct, tick_size=0.01)


def test_changed_frozen_line_is_a_new_boundary_phase_even_with_overlapping_touches():
    from caisen.main_bottoms import _same_neckline_boundary_phase

    consumed = {
        "line": {"slope": 0.0, "intercept": 1.1045},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 8, "coverage_end_index": 58}
                  for index in (8, 58)],
        "consumed_index": 80, "phase_tolerance": 0.02,
    }
    replacement = {
        "line": {"slope": 0.00015, "intercept": 1.0955},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 15, "coverage_end_index": 65}
                  for index in (15, 65)],
    }

    assert not _same_neckline_boundary_phase(consumed, replacement, tick_size=0.01)


def test_boundary_phase_equivalence_checks_projection_at_first_cross_after_shared_touches():
    from caisen.main_bottoms import _same_neckline_boundary_phase

    consumed = {
        "line": {"slope": 0.0, "intercept": 1.1045},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 8, "coverage_end_index": 58}
                  for index in (8, 58)],
        "consumed_index": 200, "phase_tolerance": 0.02,
    }
    diverging = {
        "line": {"slope": 0.0002, "intercept": 1.103},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 8, "coverage_end_index": 58}
                  for index in (8, 58)],
    }

    assert not _same_neckline_boundary_phase(consumed, diverging, tick_size=0.01)
