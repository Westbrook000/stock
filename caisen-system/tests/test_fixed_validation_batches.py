import json
import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_fixed_validation_batches import (existing_blind_codes, scan_local_candidates,
                                           select_frozen_sample, write_blind_batches,
                                           _has_current_detector_match, _preassigned_source)
from reveal_fixed_validation_answers import reveal_answers, validate_answers


def case(case_id):
    return {
        "id": case_id,
        "code": f"{600000 + int(case_id[1:]):06d}",
        "kind": "stock",
        "timeframe": "1d",
        "as_of": "2024-01-05",
        "bars": [
            ["2024-01-03", 10, 10.2, 9.8, 10.1, 100],
            ["2024-01-04", 10.1, 10.3, 9.9, 10.2, 200],
            ["2024-01-05", 10.2, 10.4, 10.0, 10.3, 300],
        ],
        "source_data_sha256": "d" * 64,
        "machine": {"label": "MACHINE_ONLY_DO_NOT_EMBED", "state": "confirmed"},
    }


def test_writes_six_answer_blind_batches_and_separate_machine_sidecar(tmp_path):
    cases = [case(f"M{number:03d}") for number in range(1, 121)]
    sidecar = {item["id"]: {"matched": ["w_bottom"], "state": "confirmed"} for item in cases}

    result = write_blind_batches(cases, sidecar, tmp_path, seed=20261029)

    assert len(result["html_files"]) == 6
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(manifest["batches"]) == 6
    assert [len(batch["case_ids"]) for batch in manifest["batches"]] == [20] * 6
    for batch in manifest["batches"]:
        html = Path(batch["html"]).read_text()
        assert "MACHINE_ONLY_DO_NOT_EMBED" not in html
        assert "pattern-blind-review.machine-answers.json" not in html
        assert "caisen-main-parent-v2" in html
        assert "caisen-v17" in html
        for shape in ("w_bottom", "spring", "bull_flag", "inverse_head_shoulders", "triangle_bottom",
                      "m_top", "false_breakout", "bear_flag", "head_shoulders", "triangle_top",
                      "other_shape", "no_book_pattern"):
            assert shape in html
        assert "overall_uncertain" in html
        assert "start_date" in html and "end_date" in html
        assert "data_fingerprint" in html
        assert "线段归属形态" in html and "lines" in html
        assert "已失效" in html and "不确定" in html
        assert "提交并导出本批评审JSON" in html
    saved_sidecar = json.loads(Path(result["machine_sidecar"]).read_text())
    assert len(saved_sidecar["cases"]) == 120
    assert saved_sidecar["cases"]["M001"]["matched"] == ["w_bottom"]
    assert saved_sidecar["main_bottom_rule_version"] == "caisen-v17-main-bottom-v1"
    assert saved_sidecar["fixed_strategy_version"] == "pattern_fixed_8_20_main_parent_v2"
    assert all("main-parent-v17" in path for path in result["html_files"])
    original_page = Path(result["html_files"][0]).read_bytes()
    try:
        write_blind_batches(cases, sidecar, tmp_path, seed=20261029)
    except FileExistsError:
        pass
    else:
        raise AssertionError("previously generated blind pages must not be overwritten")
    assert Path(result["html_files"][0]).read_bytes() == original_page


def test_batch_export_rejects_future_bars_and_duplicate_case_ids(tmp_path):
    future = case("M001")
    future["bars"].append(["2024-01-06", 10, 10, 10, 10, 100])
    cases = [future] + [case(f"M{number:03d}") for number in range(2, 121)]

    try:
        write_blind_batches(cases, {}, tmp_path, seed=20261029)
    except ValueError as exc:
        assert "after as_of" in str(exc)
    else:
        raise AssertionError("future data must not enter a frozen blind case")


def test_batch_export_rejects_duplicate_case_ids(tmp_path):
    cases = [case(f"M{number:03d}") for number in range(1, 120)]
    cases.append(case("M001"))

    try:
        write_blind_batches(cases, {item["id"]: {} for item in cases}, tmp_path, seed=20261029)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("case IDs must be unique")


def test_sample_selection_uses_union_and_independent_windows_without_reused_codes(tmp_path):
    old_review = tmp_path / "previous.html"
    old_review.write_text(
        'window.REVIEW_CASES=' + json.dumps([{"code": "000001"}]) + ';</script>',
        encoding="utf-8",
    )
    candidates = []
    number = 1
    for source in ("union", "independent"):
        for kind in ("stock", "etf"):
            for timeframe in ("1d", "1w"):
                for _ in range(16):
                    candidate = case(f"M{number:03d}")
                    candidate["code"] = f"{number:06d}"
                    candidate["kind"] = kind
                    candidate["timeframe"] = timeframe
                    candidate["sample_source"] = source
                    candidate["union_signal"] = source == "union"
                    candidate["sample_difficulty"] = "gate_diagnostic" if source == "independent" and number % 3 == 0 else "background"
                    candidates.append(candidate)
                    number += 1
    candidates[0]["code"] = "600108"

    exclusions = existing_blind_codes(old_review)
    cases, sidecar = select_frozen_sample(candidates, exclusions, seed=20261029)

    assert len(cases) == 120
    assert len({item["code"] for item in cases}) == 120
    assert not ({"000001", "600108"} & {item["code"] for item in cases})
    assert sum(sidecar[item["id"]]["sample_source"] == "union" for item in cases) == 60
    assert sum(sidecar[item["id"]]["sample_source"] == "independent" for item in cases) == 60
    assert {item["timeframe"] for item in cases} == {"1d", "1w"}
    assert {item["kind"] for item in cases} == {"stock", "etf"}
    assert sum(sidecar[item["id"]]["sample_difficulty"] == "gate_diagnostic" for item in cases) > 0


def test_union_source_ignores_ended_history_but_accepts_current_detection():
    def event(state, confirmation_date=None, selected=False, eligible=False):
        evidence = [
            {"code": "main_selection", "value": {"selected": selected, "decided_at": cutoff}},
            {"code": "main_structure", "value": {"role": "main", "signal_eligible": eligible}},
        ]
        return SimpleNamespace(pattern_type="w_bottom", direction="bullish", state=state,
                               confirmation_date=confirmation_date, evidence=evidence)

    cutoff = "2024-01-05"
    assert not _has_current_detector_match(
        [event("invalidated", "2023-12-01")], [event("invalidated", "2023-11-01")], cutoff)
    assert _has_current_detector_match([event("confirmed", cutoff)], [], cutoff)
    assert _has_current_detector_match([], [event("forming", None, selected=True)], cutoff)


def test_candidate_scanner_runs_both_detectors_at_identical_asof_prefixes(tmp_path):
    data = tmp_path / "data"
    (data / "stocks").mkdir(parents=True)
    (data / "etfs").mkdir()
    dates = __import__("pandas").date_range("2020-01-01", periods=200, freq="D")
    frame = __import__("pandas").DataFrame({
        "date": dates, "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1000,
    })
    frame.to_csv(data / "stocks" / "000002.csv", index=False)
    old_html = tmp_path / "old.html"
    old_html.write_text('window.REVIEW_CASES=[];</script>', encoding="utf-8")
    cutoffs = []

    def old_detector(window, code, kind, timeframe, *, background_bars, structure_window):
        cutoffs.append(str(background_bars.date.iloc[-1].date()))
        return [SimpleNamespace(pattern_id="p", pattern_type="w_bottom", direction="bullish",
                                state="confirmed", evidence=[])]

    def new_detector(prefix, **kwargs):
        assert str(prefix.date.iloc[-1].date()) == cutoffs[-1]
        return []

    candidates = scan_local_candidates(
        data, old_html, old_detector, new_detector,
        lambda main, legacy, **kwargs: list(main) + list(legacy), max_instruments=1,
    )

    source = _preassigned_source(20261029, "000002", "stock",
                                 "1d" if int(__import__("hashlib").sha256(
                                     b"20261029:timeframe:stock:000002").hexdigest()[:8], 16) % 2 == 0 else "1w")
    assert 1 <= len(candidates) <= 1
    assert len(cutoffs) == 1
    assert candidates[0]["sample_source"] == source
    assert all(max(bar[0] for bar in candidate["bars"]) == candidate["as_of"]
               for candidate in candidates)
    assert all(candidate["sample_source"] in {"union", "independent"} for candidate in candidates)
    assert all(candidate["union_signal"] for candidate in candidates
               if candidate["sample_source"] == "union")


def test_post_submission_reveal_checks_fingerprint_and_keeps_unknowns_distinct(tmp_path):
    bars = case("M001")["bars"]
    fingerprint = hashlib.sha256(json.dumps(bars, separators=(",", ":")).encode()).hexdigest()
    answers = {
        "validation_version": "caisen-main-parent-v2", "batch": 1,
        "cases": [{"id": f"M{number:03d}", "code": f"{600000 + number:06d}", "timeframe": "1d", "as_of": "2024-01-05",
                    "data_fingerprint": fingerprint, "chart_range": {"from": "2024-01-03", "to": "2024-01-05"},
                    "overall_uncertain": number == 1,
                    "shapes": ({"other_shape": {"state": "不确定", "start_date": "2024-01-03",
                        "end_date": "2024-01-05", "notes": "other", "lines": []}}
                        if number in (1, 2) else ({"no_book_pattern": {"state": "已确认", "start_date": "",
                        "end_date": "", "notes": "none", "lines": []}} if number == 3 else {}))}
                   for number in range(1, 21)],
    }
    machine = {"validation_version": "caisen-main-parent-v2", "cases": {
        f"M{number:03d}": {"batch": 1, "code": f"{600000 + number:06d}", "timeframe": "1d", "as_of": "2024-01-05",
                 "source_data_sha256": fingerprint, "bars": bars,
                 "linked_execution_events": ([{"pattern_type": "future_new_shape", "state": "forming",
                    "confirmation_date": None, "machine_lines": []}] if number == 3 else [])}
        for number in range(1, 21)
    }}
    answer_path, sidecar_path = tmp_path / "answers.json", tmp_path / "machine.json"
    answer_path.write_text(json.dumps(answers), encoding="utf-8")
    sidecar_path.write_text(json.dumps(machine), encoding="utf-8")

    output = reveal_answers(answer_path, sidecar_path, tmp_path / "reveal.html")
    report = Path(output).read_text(encoding="utf-8")

    assert "未知形态：'+type" in report
    assert "future_new_shape" in report
    assert "其他形态" in report
    assert "整体不确定（排除标签匹配）" in report
    assert "多标签完全一致" in report
    assert "unlabeled_cases" in report
    assert "未知机器类型排除" in report
    assert "状态不一致" in report
    answers["cases"][0]["data_fingerprint"] = "wrong"
    answer_path.write_text(json.dumps(answers), encoding="utf-8")
    try:
        validate_answers(answer_path, sidecar_path)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("answer rows must match the frozen data fingerprint")
    answers["cases"][0]["data_fingerprint"] = fingerprint
    answers["cases"][0]["shapes"] = {
        "no_book_pattern": {"state": "已确认", "start_date": "", "end_date": "", "notes": "", "lines": []},
        "w_bottom": {"state": "已确认", "start_date": "2024-01-03", "end_date": "2024-01-05", "notes": "", "lines": []},
    }
    answer_path.write_text(json.dumps(answers), encoding="utf-8")
    try:
        validate_answers(answer_path, sidecar_path)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("no_book_pattern must remain distinct from book shapes")
