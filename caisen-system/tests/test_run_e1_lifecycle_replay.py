import json
import sys
from pathlib import Path

import pytest


def _run_preflight(tmp_path, monkeypatch, members, registry):
    from caisen import daily_research
    from scripts import run_e1_lifecycle_replay as runner

    snapshot = tmp_path / "snapshot"
    for folder in ("data/stocks", "data/etfs", "caisen-system/config"):
        (snapshot / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, code in members:
        folder = "stocks" if kind == "stock" else "etfs"
        quote = snapshot / "data" / folder / f"{code}.csv"
        quote.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,1\n",
                         encoding="utf-8")
        stat = quote.stat()
        rows.append({"kind": kind, "code": code,
                     "frozen_revision": {"mtime_ns": stat.st_mtime_ns,
                                         "size_bytes": stat.st_size}})
    (snapshot / "caisen-system/config/instruments.json").write_text(
        json.dumps({"instruments": registry}), encoding="utf-8")
    monkeypatch.setenv("CAISEN_INSTRUMENTS", str(snapshot / "caisen-system/config/instruments.json"))
    (snapshot / "freeze-manifest.json").write_text(
        json.dumps({"universe_id": "fixture", "revision_policy": "mtime_size",
                    "symbols": rows, "resources": []}), encoding="utf-8")
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps({
        "id": "fixture",
        "stocks": [code for kind, code in members if kind == "stock"],
        "etfs": [code for kind, code in members if kind == "etf"],
    }), encoding="utf-8")
    output = tmp_path / "output"

    def fake_run_daily_research(*args, **kwargs):
        return {"status": "complete", "summary": {"account_count": len(members)}}

    monkeypatch.setattr(daily_research, "run_daily_research", fake_run_daily_research)
    monkeypatch.setattr(daily_research, "_all_ledger_checks_pass", lambda *_: True)
    monkeypatch.setattr(sys, "argv", [
        "run_e1_lifecycle_replay.py", "--sample", "all1000",
        "--snapshot-root", str(snapshot), "--universe", str(universe),
        "--output-dir", str(output),
    ])
    assert runner.main() == 0
    return json.loads((output / "e1-run-summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind,code", [
    ("stock", "600001"),
    ("etf", "510001"),
])
def test_preflight_accepts_stock_only_or_etf_only_cohorts(tmp_path, monkeypatch,
                                                        kind, code):
    manifest = _run_preflight(tmp_path, monkeypatch, [(kind, code)], {})

    assert manifest["cohort_counts"] == {
        "selected": 1, "stock": int(kind == "stock"), "etf": int(kind == "etf")}


def test_preflight_still_rejects_wrong_tick_for_selected_etf(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="frozen registry tick values"):
        _run_preflight(tmp_path, monkeypatch,
                       [("stock", "600001"), ("etf", "510001")],
                       {"etf:510001": {"tick_size": 0.01}})


def test_preflight_accepts_mixed_cohort_only_with_each_kind_tick(tmp_path, monkeypatch):
    manifest = _run_preflight(tmp_path, monkeypatch,
                              [("stock", "600001"), ("etf", "510001")], {})

    assert manifest["cohort_counts"] == {"selected": 2, "stock": 1, "etf": 1}


def test_selected_kind_without_resolved_ticks_is_still_rejected():
    from scripts import run_e1_lifecycle_replay as runner

    with pytest.raises(ValueError, match="etf"):
        runner._validate_selected_tick_sizes(
            [("stock", "600001"), ("etf", "510001")],
            {"stock": {0.01}, "etf": set()})
