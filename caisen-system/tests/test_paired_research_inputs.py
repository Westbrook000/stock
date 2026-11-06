import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_research_review_20261105.py"
spec = importlib.util.spec_from_file_location("research_review_inputs", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def _write_bars(path, close):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "date,open,high,low,close,volume\n"
        f"2024-01-02,{close},{close + 1},{close - 1},{close},1000\n"
        f"2024-01-03,{close},{close + 1},{close - 1},{close},1000\n",
        encoding="utf-8",
    )


def _account(path, account_revision):
    stat = path.stat()
    snapshot_revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    review.DATA_SOURCE_MAP = {
        "stock:000001": {
            "snapshot_path": str(path),
            "snapshot_root": str(path.parents[2]),
            "snapshot_data_revision": snapshot_revision,
            "snapshot_sha256": digest,
            "account_data_revision": account_revision,
        }
    }
    return {"kind": "stock", "code": "000001",
            "provenance": {"data_revision": account_revision}}


def test_period_replay_reads_the_frozen_source_map(tmp_path):
    frozen_root = tmp_path / "frozen"
    frozen_path = frozen_root / "data" / "stocks" / "000001.csv"
    _write_bars(frozen_path, 10)
    review.SNAPSHOT_ROOT = frozen_root
    account = _account(frozen_path, "run-revision:paired")

    bars = review._read_bars(account)
    assert bars.close.tolist() == [10, 10]

    # The period broker receives the mapped snapshot bars and starts with a fresh account.
    from caisen.fixed_backtest import STARTING_CASH, simulate_fixed_trades
    replay = simulate_fixed_trades(bars, [], cash=STARTING_CASH)
    assert replay["trades"] == []
    assert replay["ending_equity"] == STARTING_CASH


def test_paired_account_revision_must_match_source_map(tmp_path):
    frozen_path = tmp_path / "frozen" / "data" / "stocks" / "000001.csv"
    _write_bars(frozen_path, 10)
    review.SNAPSHOT_ROOT = frozen_path.parents[2]
    account = _account(frozen_path, "revision:baseline")
    account["provenance"]["data_revision"] = "revision:min3-mismatch"

    with pytest.raises(RuntimeError, match="account data revision does not match"):
        review._read_bars(account)


def test_promotion_table_applies_locked_validation_and_retention_screen():
    payload = {"periods": []}
    for name in ("through_2020", "2021_2023", "2024_2026"):
        variants = {}
        for variant in ("baseline", "open_admission", "neckline_min3", "neckline_min3_open_admission"):
            variants[variant] = {"trade_count": 100,
                                 "metrics": {"expectancy": 10.0, "maximum_drawdown": 0.10}}
        payload["periods"].append({"name": name, "variants": variants})
    for row in payload["periods"]:
        candidate = row["variants"]["open_admission"]
        candidate["metrics"]["expectancy"] = 11.0
        candidate["metrics"]["maximum_drawdown"] = 0.09
    payload["periods"][0]["variants"]["open_admission"]["metrics"]["expectancy"] = 8.0
    payload["periods"][2]["variants"]["open_admission"]["trade_count"] = 49

    rendered = review._promotion_table_html(payload)
    assert "未达到研究门槛" in rendered
    assert "开发期下降" in rendered
    assert "生产基线" in rendered

    payload["periods"][2]["variants"]["open_admission"]["trade_count"] = 50
    rendered = review._promotion_table_html(payload)
    assert "具备进一步研究资格" in rendered


def test_recross_audit_rejects_other_or_unfrozen_neckline_candidates():
    pattern = {
        "pattern_id": "p1", "pattern_type": "w_bottom", "formed_at": "2025-12-16",
        "evidence": [
            {"code": "main_structure", "value": {"maturity_date": "2026-06-16",
                                                     "neckline": {"slope": 0.0, "intercept": 1.1}}},
            {"code": "neckline_selection", "value": {
                "selected_candidate_id": "selected",
                "structure_known_at": "2026-04-16", "line_frozen_at": "2026-03-19",
                "selected_line": {"slope": 0.0, "intercept": 1.1},
                "early_breakouts": [
                    {"candidate_id": "other", "date": "2026-05-01", "index": 10,
                     "frozen_neckline": 1.1},
                    {"candidate_id": "selected", "date": "2026-04-10", "index": 10,
                     "frozen_neckline": 1.1},
                    {"candidate_id": "selected", "date": "2026-05-05", "index": 10,
                     "frozen_neckline": 1.2},
                    {"candidate_id": "selected", "date": "2026-05-06", "index": 10,
                     "frozen_neckline": 1.1},
                ],
            }},
        ],
    }
    diagnostics = Counter()
    rows = review._early_breakout_late_recross_rows(
        {"kind": "etf", "code": "159122",
         "signals": [{"signal_date": "2026-06-22", "patterns": [pattern]}]}, diagnostics)

    assert len(rows) == 1
    assert rows[0]["first_pre_maturity_buffered_cross"]["date"] == "2026-05-06"
    assert dict(diagnostics) == {
        "other_boundary_retired": 1,
        "before_structure_known_or_line_frozen": 1,
        "selected_line_projection_mismatch": 1,
    }
