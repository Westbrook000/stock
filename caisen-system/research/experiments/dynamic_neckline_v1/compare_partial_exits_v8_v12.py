"""Fair execution sensitivity: replay saved V8 and V12 signals with one exit rule.

This does not assess recognition accuracy and does not modify production strategy.
Both signal sets use the same 100-stock cohort, frozen OHLCV snapshot, and cutoff.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(ROOT))

from caisen.fixed_backtest import calculate_metrics  # noqa: E402
from run_v2_position_fraction import replay  # noqa: E402

WORKSPACE = ROOT.parent
REPORTS = ROOT / "reports" / "daily-research"
COHORT = REPORTS / "full-unified-v12-20261106-w4-rerun" / "partial-100-report"
V8_ACCOUNTS = REPORTS / "full-20261104-v8-thread2-boundary" / "accounts"
V12_ACCOUNTS = COHORT / "accounts"
DATA = REPORTS / "paired-input-snapshot-20261105-v1" / "data" / "stocks"
OUTPUT = (EXPERIMENT_DIR / "results" / "v8-v12-partial-exit-comparison-20261111.json")
CUTOFF = "2026-10-30"
STARTING_CASH = 100_000.0


def _summarize(accounts: list[dict]) -> dict:
    trades = [trade for account in accounts for trade in account["trades"]]
    starting = len(accounts) * STARTING_CASH
    ending = sum(float(account["ending_equity"]) for account in accounts)
    portfolio_dates = sorted({point["date"] for account in accounts
                              for point in account["daily_equity"]})
    portfolio_curve = []
    account_curves = [{point["date"]: point for point in account["daily_equity"]}
                      for account in accounts]
    carried = [{"equity": STARTING_CASH, "invested_value": 0.0}
               for _ in accounts]
    for date in portfolio_dates:
        points = []
        for index, curve in enumerate(account_curves):
            if date in curve:
                carried[index] = curve[date]
            points.append(carried[index])
        portfolio_curve.append({
            "date": date,
            "equity": sum(point["equity"] for point in points),
            "invested_value": sum(point["invested_value"] for point in points),
        })
    metrics = calculate_metrics(trades, starting, ending, portfolio_curve)
    account_returns = [account["ending_equity"] / STARTING_CASH - 1 for account in accounts]
    metrics.update({
        "accounts_positive": sum(value > 0 for value in account_returns),
        "median_account_return": sorted(account_returns)[len(account_returns) // 2],
        "accounting_all_pass": all(all(account["accounting_checks"].values())
                                    for account in accounts),
        "account_count": len(accounts),
    })
    return metrics


def _run_side(label: str, account_dir: Path, codes: list[str]) -> dict:
    accounts = []
    for index, code in enumerate(codes, 1):
        source = account_dir / f"stock-{code}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        saved = json.loads(source.read_text(encoding="utf-8"))
        if saved.get("code") != code or saved.get("status") != "complete":
            raise ValueError(f"invalid saved account {source}")
        bars = pd.read_csv(DATA / f"{code}.csv")
        bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
        bars = bars.loc[bars.date <= CUTOFF].reset_index(drop=True)
        result = replay(bars, saved.get("signals", []), allocation_fraction=1.0,
                        cash=STARTING_CASH, lot_size=100)
        if not all(result["accounting_checks"].values()):
            raise AssertionError(f"{label}/{code} accounting failed: {result['accounting_checks']}")
        result["code"] = code
        result["metrics"] = calculate_metrics(result["trades"], STARTING_CASH,
                                              result["ending_equity"], result["daily_equity"])
        result["source_signal_count"] = len(saved.get("signals", []))
        accounts.append(result)
        if index % 20 == 0:
            print(f"{label}: {index}/{len(codes)}", flush=True)
    return {"summary": _summarize(accounts), "accounts": accounts}


def main() -> None:
    selection = json.loads((COHORT / "sample-selection.json").read_text(encoding="utf-8"))
    baseline = json.loads((REPORTS / "paired-100-v8-v12-20261109" / "comparison.json")
                          .read_text(encoding="utf-8"))
    codes = sorted(str(item["code"]).zfill(6) for item in selection["symbols"]
                   if item["kind"] == "stock")
    if len(codes) != 100 or len(set(codes)) != 100:
        raise ValueError(f"expected 100 unique stocks, got {len(codes)}")
    comparison_codes = {str(item["code"]).zfill(6) for item in baseline.get("sample", [])}
    if set(codes) != comparison_codes:
        raise ValueError("selected 100-stock cohort differs from the saved V8/V12 comparison")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    missing = [code for code in codes if not (DATA / f"{code}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen snapshot files: {missing[:10]}")

    started = time.perf_counter()
    v8 = _run_side("V8", V8_ACCOUNTS, codes)
    v12 = _run_side("V12", V12_ACCOUNTS, codes)
    payload = {
        "experiment": "same_execution_partial_exit_v8_v12",
        "scope": "saved V8/V12 signals replayed on common frozen data; not a recognition comparison",
        "cohort_count": len(codes), "codes": codes, "cutoff": CUTOFF,
        "exit_rule": "60% at +8%; remaining shares at -8% stop or +20% target; zero costs; no max hold",
        "data_snapshot": str(DATA), "starting_cash_per_account": STARTING_CASH,
        "baseline_reference": {
            "v8": baseline["versions"]["old"]["summary"],
            "v12": baseline["versions"]["new"]["summary"],
        },
        "v8": v8, "v12": v12,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "historical development cohort, not an independent holdout",
            "execution variant is compared symmetrically; it does not validate recognition accuracy",
            "saved signals are replayed without redetecting structures",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "elapsed_seconds": payload["elapsed_seconds"],
                      "v8": v8["summary"], "v12": v12["summary"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
