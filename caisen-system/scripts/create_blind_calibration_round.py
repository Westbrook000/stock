"""Build a human chart-review round from current local CSV history."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "caisen-system"
sys.path.insert(0, str(PROJECT))

from caisen.calibration_pack import build_blind_cases, render_review_html


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20, help="number of charts to sample (default: 20)")
    parser.add_argument("--seed", type=int, default=20261025, help="fixed random seed for reproducible sampling")
    parser.add_argument("--output", type=Path, default=PROJECT / "calibration" / "round_02.html")
    args = parser.parse_args(argv)
    cases = build_blind_cases(ROOT, count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_review_html(cases, seed=args.seed), encoding="utf-8")
    print("review page:", args.output)
    print("cases:", len(cases))
    for kind in ("stock", "etf"):
        for timeframe in ("1d", "1w"):
            n = sum(case["kind"] == kind and case["timeframe"] == timeframe for case in cases)
            print("%s %s: %d" % (kind, timeframe, n))
    print("Machine pattern predictions were not used to sample or render cases.")


if __name__ == "__main__":
    main()
