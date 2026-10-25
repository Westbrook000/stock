"""Compare the user's ten chart labels with current machine pattern snapshots."""

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "caisen-system"
sys.path.insert(0, str(PROJECT))

from caisen.calibration import compare_labels, load_labels
from caisen import RULE_VERSION
from caisen.market import load_local_history
from caisen.market import aggregate_weekly, default_china_calendar
from caisen.patterns import current_pattern_snapshot, detect_patterns
from caisen.rules.catalog import load_rule_catalog

TERMINAL_STATES = {"invalidated", "target_reached", "followthrough_confirmed"}


def evaluate(labels_path, output_path, clarifications_path=None, root=ROOT):
    labels = load_labels(labels_path, clarifications_path=clarifications_path)
    label_metadata = ({
        "calibration_round": labels[0].get("calibration_round"),
        "calibration_pack_version": labels[0].get("calibration_pack_version"),
        "label_engine_version": labels[0].get("label_engine_version"),
        "label_source_rule_version": labels[0].get("label_source_rule_version"),
    } if labels else {})
    cases = []
    machine = []
    for label in labels:
        kind = label.get("kind", "stock")
        timeframe = label.get("timeframe", "1d")
        bars = load_local_history(label["code"], kind, root=str(root))
        if bars is None:
            found = []
            data_error = "local history file missing"
        else:
            bars = bars[bars.date <= label["asof"]].reset_index(drop=True)
            if timeframe == "1w":
                bars = aggregate_weekly(bars, asof=label["asof"], calendar=default_china_calendar())
            if bars.empty:
                found = []
                data_error = "no bars available at label date"
            else:
                found = current_pattern_snapshot(detect_patterns(
                    bars, code=label["code"], kind=kind, timeframe=timeframe, asof=label["asof"]
                ))
                data_error = None
        machine.extend(pattern.to_dict() for pattern in found)
        serialized = [pattern.to_dict() for pattern in found]
        cases.append({
            "case_id": label["case_id"],
            "code": label["code"],
            "name": label["name"],
            "kind": kind,
            "timeframe": timeframe,
            "asof": label["asof"],
            "user_label": {"pattern_type": label["pattern_type"], "state": label["state"], "display": label["display_label"]},
            "machine_snapshot": [{"pattern_type": pattern.pattern_type, "state": pattern.state,
                                  "formed_at": pattern.formed_at, "confirmed_at": pattern.confirmed_at,
                                  "evidence": pattern.evidence} for pattern in found],
            "active_machine_snapshot": [item for item in serialized if item["state"] not in TERMINAL_STATES],
            "terminal_history_snapshot": [item for item in serialized if item["state"] in TERMINAL_STATES],
            "data_error": data_error,
        })
    report = compare_labels(labels, machine)
    payload = {
        "label_source": str(Path(labels_path).name),
        "clarifications_source": str(Path(clarifications_path).name) if clarifications_path else None,
        "data_basis": "current_local_snapshot_truncated_to_user_asof",
        "rule_version": RULE_VERSION,
        "source_rule_version": load_rule_catalog()["rule_set_version"],
        "label_metadata": label_metadata,
        "version_comparison": "label provenance and evaluation rules are stored separately; reports are not pooled across rounds or rule versions",
        "interpretation": "manual_review_required_not_accuracy_or_return_validation",
        "summary": report,
        "cases": cases,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=PROJECT / "calibration" / "labels" / "round_01_v2.user.json")
    parser.add_argument("--clarifications", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "calibration" / "reports" / "round_01_v15.machine-review.json")
    args = parser.parse_args()
    labels_file = args.labels
    clarifications_file = args.clarifications
    if clarifications_file is None and labels_file.name == "round_01_v2.user.json":
        candidate = labels_file.with_name("round_01_v2.clarifications.json")
        clarifications_file = candidate if candidate.exists() else None
    output_file = args.output
    result = evaluate(labels_file, output_file,
                      clarifications_path=clarifications_file if clarifications_file and clarifications_file.exists() else None)
    summary = result["summary"]
    print("report:", output_file)
    print("pattern agreement:", summary["pattern_agreement"], "/", summary["total"])
    print("state agreement:", summary["state_agreement"], "/", summary["state_total"], "labeled patterns")
    print("disagreements:", len(summary["disagreements"]))
