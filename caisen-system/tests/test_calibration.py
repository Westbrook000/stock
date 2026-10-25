import json
import importlib.util
from pathlib import Path

import pandas as pd

from caisen.calibration import load_labels, compare_labels


def test_loads_user_round_labels_without_collapsing_forming_state(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({
        "1": {"code": "000001", "name": "样本", "asof": "2026-10-16", "pattern": "W", "state": "forming", "note": ""},
        "2": {"code": "000002", "name": "样本二", "asof": "2026-10-16", "pattern": "N", "state": "", "note": ""},
    }, ensure_ascii=False), encoding="utf-8")

    labels = load_labels(path)

    assert labels[0]["pattern_type"] == "w_bottom"
    assert labels[0]["state"] == "forming"
    assert labels[0]["display_label"] == "W底，未确认"
    assert labels[1]["pattern_type"] is None
    assert labels[1]["display_label"] == "没有任何书本形态"


def test_load_labels_applies_separate_user_clarification_without_overwriting_source(tmp_path):
    source = tmp_path / "answers.json"
    corrections = tmp_path / "clarifications.json"
    original = {"10": {"code": "688216", "name": "气派科技", "asof": "2026-10-16",
                       "pattern": "HS", "state": "forming", "note": ""}}
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    corrections.write_text(json.dumps({"10": {"pattern": "N", "state": "none",
                                               "note": "无任何书本形态",
                                               "clarified_at": "2026-10-25"}}, ensure_ascii=False),
                           encoding="utf-8")

    labels = load_labels(source, clarifications_path=corrections)

    assert labels[0]["pattern_type"] is None
    assert labels[0]["state"] == "none"
    assert labels[0]["label_source"] == "user_clarification"
    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_comparison_reports_machine_disagreements_without_claiming_accuracy():
    labels = [{"code": "000001", "asof": "2026-10-16", "pattern_type": "w_bottom", "state": "forming"}]
    machine = [{"code": "000001", "pattern_type": "w_bottom", "state": "confirmed"}]

    report = compare_labels(labels, machine)

    assert report["total"] == 1
    assert report["pattern_agreement"] == 1
    assert report["state_agreement"] == 0
    assert report["state_total"] == 1
    assert report["disagreements"][0]["field"] == "state"
    assert "当前活动十招均无成立中的形态" in report["none_label_semantics"]
    assert report["interpretation"] == "calibration_disagreements_only"


def test_confirmed_user_label_matches_terminal_machine_state_after_confirmation():
    labels = [{"code": "000001", "pattern_type": "w_bottom", "state": "confirmed"}]
    machine = [{"code": "000001", "pattern_type": "w_bottom", "state": "target_reached",
                "evidence": [{"code": "close_confirmation", "value": True}]}]

    report = compare_labels(labels, machine)

    assert report["pattern_agreement"] == 1
    assert report["state_agreement"] == 1


def test_forming_user_label_does_not_match_candidate_invalidated_before_confirmation():
    labels = [{"code": "000001", "pattern_type": "head_shoulders", "state": "forming"}]
    machine = [{"code": "000001", "pattern_type": "head_shoulders", "state": "invalidated",
                "evidence": [{"code": "close_confirmation", "value": False}]}]

    report = compare_labels(labels, machine)

    assert report["pattern_agreement"] == 1
    assert report["state_agreement"] == 0
    assert report["disagreements"][0]["field"] == "state"


def test_none_label_ignores_terminal_history_but_not_active_candidates():
    labels = [{"case_id": "none", "code": "000001", "pattern_type": None, "state": "none"}]

    history_only = compare_labels(labels, [
        {"code": "000001", "pattern_type": "w_bottom", "state": "invalidated"},
        {"code": "000001", "pattern_type": "m_top", "state": "target_reached"},
    ])
    assert history_only["pattern_agreement"] == 1
    assert history_only["state_total"] == 0

    active_candidate = compare_labels(labels, [
        {"code": "000001", "pattern_type": "w_bottom", "state": "forming"},
        {"code": "000001", "pattern_type": "w_bottom", "state": "invalidated"},
    ])
    assert active_candidate["pattern_agreement"] == 0
    assert active_candidate["disagreements"][0]["machine"] == ["w_bottom"]


def test_imported_round_two_has_ten_user_labels_and_forming_examples():
    path = Path(__file__).parents[1] / "calibration" / "labels" / "round_01_v2.user.json"

    labels = load_labels(path)

    assert len(labels) == 10
    assert sum(label["state"] == "forming" for label in labels) == 2
    assert labels[2]["display_label"] == "W底，未确认"


def test_round_one_metadata_is_resolved_without_changing_the_source_labels():
    path = Path(__file__).parents[1] / "calibration" / "labels" / "round_01_v2.user.json"

    labels = load_labels(path)

    assert all(label["calibration_round"] == "round_01_v2" for label in labels)
    assert all(label["label_engine_version"] == "caisen-v14" for label in labels)
    assert all(label["label_source_rule_version"] == "caisen-rules-v5" for label in labels)


def test_historical_v14_report_keeps_its_original_rule_versions():
    path = Path(__file__).parents[1] / "calibration" / "reports" / "round_01_v14.machine-review.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["rule_version"] == "caisen-v14"
    assert report["source_rule_version"] == "caisen-rules-v5"


def test_new_evaluation_report_keeps_label_and_engine_versions_separate(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "evaluate_calibration_labels.py"
    spec = importlib.util.spec_from_file_location("caisen_evaluator_test", script)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    labels_path = tmp_path / "answers.json"
    labels_path.write_text(json.dumps({
        "_meta": {"calibration_round": "round_02", "calibration_pack_version": "legacy-v1",
                  "engine_version": "caisen-v14", "source_rule_version": "caisen-rules-v5"},
        "case-1": {"code": "000001", "pattern": "W", "state": "forming", "asof": "2026-10-23"},
    }), encoding="utf-8")
    monkeypatch.setattr(evaluator, "load_local_history", lambda code, kind, root: pd.DataFrame({
        "date": [pd.Timestamp("2026-10-23")],
    }))
    monkeypatch.setattr(evaluator, "detect_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluator, "current_pattern_snapshot", lambda patterns: [])
    output = tmp_path / "round_02_v15.machine-review.json"

    report = evaluator.evaluate(labels_path, output, root=tmp_path)

    assert report["rule_version"] == "caisen-v15"
    assert report["source_rule_version"] == "caisen-rules-v6"
    assert report["label_metadata"] == {
        "calibration_round": "round_02", "calibration_pack_version": "legacy-v1",
        "label_engine_version": "caisen-v14", "label_source_rule_version": "caisen-rules-v5",
    }
    assert "not pooled across rounds" in report["version_comparison"]
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_round_labels_preserve_instrument_kind_and_timeframe(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({
        "blind-1": {"code": "510300", "name": "ETF", "asof": "2026-10-23",
                    "pattern": "W", "state": "forming", "kind": "etf", "timeframe": "1w"}
    }, ensure_ascii=False), encoding="utf-8")

    label = load_labels(path)[0]

    assert label["kind"] == "etf"
    assert label["timeframe"] == "1w"


def test_legacy_composite_labels_remain_readable_with_their_version_metadata(tmp_path):
    path = tmp_path / "legacy-answers.json"
    path.write_text(json.dumps({
        "_meta": {"calibration_round": "round_02", "engine_version": "caisen-v14",
                  "source_rule_version": "caisen-rules-v5"},
        "legacy-ws": {"code": "000001", "pattern": "WS", "state": "forming"},
        "legacy-fbhs": {"code": "000002", "pattern": "FBHS", "state": "confirmed"},
    }), encoding="utf-8")

    labels = load_labels(path)

    assert [label["pattern_type"] for label in labels] == ["w_bottom_spring", "false_breakout_head_shoulders"]
    assert all(label["label_engine_version"] == "caisen-v14" for label in labels)
    assert all(label["label_source_rule_version"] == "caisen-rules-v5" for label in labels)
    assert all(label["calibration_round"] == "round_02" for label in labels)


def test_comparison_does_not_match_same_code_pattern_from_another_timeframe():
    labels = [{"case_id": "weekly", "code": "000001", "kind": "stock", "timeframe": "1w",
               "pattern_type": "w_bottom", "state": "forming"}]

    report = compare_labels(labels, [
        {"code": "000001", "kind": "stock", "timeframe": "1d", "pattern_type": "w_bottom", "state": "forming"}
    ])

    assert report["pattern_agreement"] == 0
    assert report["disagreements"][0]["machine"] == []
