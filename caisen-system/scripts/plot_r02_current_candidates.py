"""Plot the frozen R02 000008 forming W and representative triangle as PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SESSION_DEFAULT = ROOT / "reports/daily-research/structure-boundary-study-20261112/g1-prefix-causality/R02-000008-session-snapshot.json"
SNAPSHOT_DEFAULT = Path("/tmp/caisen-structure-boundary-snapshot-20261112-v2")
OUTPUT_DEFAULT = ROOT / "study/R02-current-candidates"
W_DATES = ("2024-10-18", "2024-11-27")
CHART_FROM = "2024-01-30"


def _bars(csv_path: Path, as_of: str) -> pd.DataFrame:
    bars = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["date"])
    for col in ("open", "high", "low", "close"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    bars = (bars.dropna(subset=["date", "open", "high", "low", "close"])
            .sort_values("date", kind="mergesort")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True))
    bars = bars[bars.date.dt.strftime("%Y-%m-%d") <= as_of].reset_index(drop=True)
    bars["full_history_index"] = range(len(bars))
    return bars


def _select_w(structures: list[dict]) -> dict:
    matches = [x for x in structures if x.get("pattern_type") == "w_bottom"
               and (x.get("key_points", {}).get("pivot_1_date"),
                    x.get("key_points", {}).get("pivot_2_date")) == W_DATES]
    if len(matches) != 1:
        raise ValueError(f"expected one forming W with lows {W_DATES}, found {len(matches)}")
    return matches[0]


def _has_explicit_primary(pattern: dict) -> bool:
    points = pattern.get("key_points", {})
    lifecycle = pattern.get("joint_lifecycle", {})
    return bool(points.get("role") == "main" or
                (lifecycle.get("selection_at_confirmation") or {}).get("selected") is True)


def _select_triangle(structures: list[dict]) -> tuple[dict, str, int]:
    triangles = [x for x in structures if x.get("pattern_type") == "triangle_bottom"]
    if not triangles:
        raise ValueError("no forming triangle candidates in session snapshot")
    primary = [x for x in triangles if _has_explicit_primary(x)]
    if len(primary) == 1:
        item = primary[0]
        kp = item["key_points"]
        return item, "explicit main role/selection present in session snapshot", kp["latest_anchor_index"] - kp["first_trough_index"]
    if len(primary) > 1:
        raise ValueError(f"multiple explicit primary triangles: {len(primary)}")
    item = max(triangles, key=lambda x: (
        int(x["key_points"]["latest_anchor_index"]) - int(x["key_points"]["first_trough_index"]),
        str(x.get("pattern_id", ""))))
    kp = item["key_points"]
    return item, "no explicit primary role/selection is recorded; selected the candidate with the longest full-history anchor span", int(kp["latest_anchor_index"]) - int(kp["first_trough_index"])


def _line_dates(pattern: dict) -> tuple[str, str, str]:
    lifecycle = pattern.get("joint_lifecycle", {})
    tests = lifecycle.get("boundary_anchor_tests", [])
    if not tests:
        raise ValueError(f"{pattern.get('pattern_id')} has no recorded boundary tests")
    # This prefix snapshot does not expose neckline_selection.effective_start_date.
    # Use the first recorded frozen boundary test as the plotted effective start.
    display_start = min(str(test["pivot_date"]) for test in tests)
    freeze = str(lifecycle.get("paired_at") or "")
    as_of = str(pattern.get("last_observed_date") or "")
    if not (display_start <= freeze <= as_of):
        raise ValueError(f"invalid boundary date order: {display_start}, {freeze}, {as_of}")
    return display_start, freeze, as_of


def _verify_indices(pattern: dict, date_to_index: dict[str, int]) -> None:
    kp = pattern["key_points"]
    for n in range(1, 6):
        date, stored = kp.get(f"pivot_{n}_date"), kp.get(f"pivot_{n}_index")
        if date and stored is not None and date_to_index.get(date) != int(stored):
            raise ValueError(f"stored pivot index mismatch {pattern['pattern_id']} {date}: {stored} != {date_to_index.get(date)}")


def _draw_candles(ax, bars: pd.DataFrame, date_to_index: dict[str, int], start: str,
                  end: str) -> pd.DataFrame:
    view = bars[(bars.date >= pd.Timestamp(start)) & (bars.date <= pd.Timestamp(end))]
    if view.empty:
        raise ValueError(f"empty chart range {start} through {end}")
    width = 0.58
    for row in view.itertuples():
        x = mdates.date2num(row.date)
        color = "#c94c43" if row.close >= row.open else "#21856b"
        ax.vlines(x, row.low, row.high, color=color, linewidth=0.65, zorder=2)
        body = max(abs(row.close - row.open), 0.003)
        ax.add_patch(Rectangle((x - width / 2, min(row.open, row.close)), width, body,
                               facecolor=color, edgecolor=color, linewidth=0.35, zorder=3))
    ax.plot(view.date, view.close, color="#334a63", linewidth=0.45, alpha=0.35, zorder=1)
    return view


def _draw_pattern(ax, bars: pd.DataFrame, date_to_index: dict[str, int], pattern: dict,
                  color: str, prefix: str, title: str) -> dict:
    kp = pattern["key_points"]
    lifecycle = pattern["joint_lifecycle"]
    start, freeze, as_of = _line_dates(pattern)
    index_to_date = {i: d for d, i in date_to_index.items()}
    i_start, i_freeze, i_asof = (date_to_index[start], date_to_index[freeze], date_to_index[as_of])
    slope, intercept = float(pattern["neckline"]["slope"]), float(pattern["neckline"]["intercept"])

    solid_indices = list(range(i_start, i_freeze + 1))
    future_indices = list(range(i_freeze, i_asof + 1))
    ax.plot([pd.Timestamp(index_to_date[i]) for i in solid_indices],
            [slope * i + intercept for i in solid_indices], color=color, linewidth=2.0,
            linestyle="-", label=f"Frozen line: {start} to paired {freeze}", zorder=4)
    ax.plot([pd.Timestamp(index_to_date[i]) for i in future_indices],
            [slope * i + intercept for i in future_indices], color=color, linewidth=1.8,
            linestyle=(0, (4, 3)), alpha=0.9, label=f"Frozen extension to as-of {as_of}", zorder=4)
    ax.axvline(pd.Timestamp(freeze), color=color, linewidth=1.0, alpha=0.65, linestyle=":", zorder=2)
    ax.axvline(pd.Timestamp(as_of), color="#465363", linewidth=0.9, alpha=0.55, linestyle=":", zorder=2)
    ax.scatter(pd.Timestamp(as_of), slope * i_asof + intercept, marker="o", s=23,
               facecolor="white", edgecolor=color, linewidth=1.3, zorder=6)

    limit = 2 if pattern["pattern_type"] == "w_bottom" else 5
    test_number_by_date = {str(t["pivot_date"]): i for i, t in enumerate(lifecycle.get("boundary_anchor_tests", []), 1)}
    anchors = []
    for n in range(1, limit + 1):
        date, price, kind = kp.get(f"pivot_{n}_date"), kp.get(f"pivot_{n}"), kp.get(f"pivot_{n}_kind")
        if date is None or price is None:
            continue
        _verify_indices(pattern, date_to_index)
        marker = "o" if kind == "low" else "^"
        ax.scatter(pd.Timestamp(date), float(price), marker=marker, s=55, color=color,
                   edgecolor="white", linewidth=0.9, zorder=7)
        matching_test = test_number_by_date.get(date)
        suffix = f" · boundary test {matching_test}" if matching_test else ""
        label = f"{prefix}{n} {kind} {date}  {float(price):.3f}{suffix}"
        if pattern["pattern_type"] == "w_bottom":
            offset = (-138, 22) if n == 1 else (14, -27)
        else:
            offset = (7, 10 if n % 2 else -17)
        ax.annotate(label, (pd.Timestamp(date), float(price)), xytext=offset,
                    textcoords="offset points", fontsize=7.5, color=color,
                    bbox={"boxstyle": "round,pad=.18", "fc": "white", "ec": color, "alpha": .88})
        anchors.append({"number": n, "date": date, "kind": kind, "price": float(price),
                        "full_history_index": int(date_to_index[date])})

    tests = []
    for n, test in enumerate(lifecycle.get("boundary_anchor_tests", []), 1):
        date, level = str(test["pivot_date"]), float(test["level"])
        ax.scatter(pd.Timestamp(date), level, marker="D", s=43, color="#232f3e",
                   edgecolor="white", linewidth=0.8, zorder=8)
        overlaps_anchor = any(kp.get(f"pivot_{j}_date") == date for j in range(1, limit + 1))
        if not overlaps_anchor:
            label = f"Test {n} {date}  {level:.3f}\nknown_at line freeze {freeze}"
            ax.annotate(label, (pd.Timestamp(date), level), xytext=(7, 10 if n == 1 else -28),
                        textcoords="offset points", fontsize=7.2, color="#232f3e",
                        bbox={"boxstyle": "round,pad=.2", "fc": "#fff9e9", "ec": "#b8a56e", "alpha": .93})
        tests.append({"date": date, "level": level, "group_id": test.get("group_id"),
                      "known_at_line_freeze": freeze,
                      "full_history_index": int(date_to_index[date])})

    ax.text(0.012, 0.975,
            f"State: {pattern['state']}  |  entry_eligible: {lifecycle.get('entry_eligible')}\n"
            f"Boundary paired/frozen; known_at: {freeze}  |  as_of: {as_of}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#adb9c5", "alpha": .94})
    ax.set_title(title, loc="left", weight="bold", fontsize=11)
    ax.grid(axis="y", color="#d8e0e7", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("Frozen CSV OHLC price")
    ax.legend(loc="upper right", ncols=1, framealpha=0.92, fontsize=7.2)
    return {"pattern_id": pattern["pattern_id"], "pattern_type": pattern["pattern_type"],
            "state": pattern["state"], "formed_at": pattern.get("formed_at"),
            "confirmation_date": pattern.get("confirmation_date"),
            "entry_eligible": lifecycle.get("entry_eligible"),
            "structure_span_bars": int(kp["latest_anchor_index"]) - int(kp["first_trough_index"]),
            "structure_start": kp["first_trough_date"], "latest_anchor_date": kp["latest_anchor_date"],
            "anchors": anchors, "boundary_tests": tests,
            "boundary": {"slope": slope, "intercept": intercept,
                         "display_effective_start": start,
                         "display_start_rule": "earliest recorded boundary_anchor_tests pivot_date; snapshot does not expose neckline_selection.effective_start_date",
                         "paired_at_freeze_date": freeze, "known_at": freeze,
                         "as_of_extension_end": as_of,
                         "start_index": int(i_start), "freeze_index": int(i_freeze),
                         "as_of_index": int(i_asof),
                         "value_at_start": slope * i_start + intercept,
                         "value_at_freeze": slope * i_freeze + intercept,
                         "value_at_as_of": slope * i_asof + intercept}}


def render(session_path: Path, bars_path: Path, output_dir: Path) -> tuple[Path, Path]:
    session = json.loads(session_path.read_text())
    as_of = str(session["as_of"])
    bars = _bars(bars_path, as_of)
    if len(bars) != int(session["input_rows"]):
        raise ValueError(f"frozen bar count mismatch: {len(bars)} != {session['input_rows']}")
    date_to_index = {d.date().isoformat(): int(i) for d, i in zip(bars.date, bars.full_history_index)}
    if date_to_index.get(as_of) != len(bars) - 1:
        raise ValueError("as-of bar is not the final row of the causal input prefix")
    structures = session["current_forming_or_confirmed_structures"]
    w = _select_w(structures)
    triangle, triangle_selection, triangle_span = _select_triangle(structures)
    triangle_count = sum(x.get("pattern_type") == "triangle_bottom" for x in structures)
    if triangle_count != 34:
        raise ValueError(f"expected 34 overlapping triangle candidates in this snapshot, found {triangle_count}")
    _verify_indices(w, date_to_index)
    _verify_indices(triangle, date_to_index)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=180, sharex=False)
    fig.patch.set_facecolor("#fbfcfe")
    for ax in axes:
        ax.set_facecolor("#fbfcfe")
    view_w = _draw_candles(axes[0], bars, date_to_index, CHART_FROM, as_of)
    w_detail = _draw_pattern(axes[0], bars, date_to_index, w, "#7551a3", "W",
        "Panel 1 · unique forming W · lows 2024-10-18 / 2024-11-27")
    axes[0].set_xlim(pd.Timestamp(CHART_FROM), pd.Timestamp(as_of) + pd.Timedelta(days=12))
    axes[0].set_ylim(float(view_w.low.min()) * 0.96, float(view_w.high.max()) * 1.04)

    view_t = _draw_candles(axes[1], bars, date_to_index, CHART_FROM, as_of)
    triangle_detail = _draw_pattern(axes[1], bars, date_to_index, triangle, "#c17c18", "T",
        f"Panel 2 · representative triangle · {triangle_span} bars · {triangle['pattern_id']}")
    axes[1].set_xlim(pd.Timestamp(CHART_FROM), pd.Timestamp(as_of) + pd.Timedelta(days=12))
    axes[1].set_ylim(float(view_t.low.min()) * 0.96, float(view_t.high.max()) * 1.04)
    axes[1].set_xlabel("Trading date")
    fig.suptitle("000008 · R02 frozen current-forming candidates · as of 2026-03-30",
                 x=0.055, y=0.995, ha="left", fontsize=14, weight="bold")
    fig.text(0.055, 0.005,
        "Frozen CSV candles; boundaries are evaluated with the original full-history row index. Solid = first recorded boundary test to paired/freeze date; dashed = finite frozen-line extension through as-of. "
        "34 overlapping triangle variants exist; only the representative longest-span candidate is drawn. Human label: none. This candidate plot does not validate recognition or infer a buy signal.",
        ha="left", va="bottom", fontsize=7.6, color="#526170", wrap=True)
    fig.tight_layout(rect=(0.04, 0.035, 0.99, 0.97), h_pad=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "R02-000008-current-candidates.png"
    json_path = output_dir / "R02-000008-current-candidates.json"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    metadata = {
        "source_session": str(session_path), "source_csv": str(bars_path),
        "snapshot_cutoff": as_of, "chart_from": CHART_FROM,
        "ohlc_rows_through_as_of": len(bars), "as_of_full_history_index": date_to_index[as_of],
        "tick_size": session.get("tick_size"), "tick_source": session.get("tick_source"),
        "candidate_counts": {"all_current_forming_or_confirmed": len(structures),
                              "w_bottom": sum(x.get("pattern_type") == "w_bottom" for x in structures),
                              "triangle_bottom": triangle_count,
                              "triangle_variants_drawn": 1},
        "w_selection": "unique forming W whose pivot dates are 2024-10-18 and 2024-11-27",
        "triangle_selection": triangle_selection,
        "human_label": "none",
        "limit": "A no-signal snapshot and a rendered candidate do not prove recognition correctness or entry eligibility.",
        "boundary_date_semantics": "The session snapshot provides joint_lifecycle.paired_at and boundary_anchor_tests, but not neckline_selection.effective_start_date or test known_at timestamps. The plotted display start is the earliest recorded boundary test; paired_at is treated as the frozen-line known_at.",
        "patterns": {"w_bottom": w_detail, "triangle_bottom": triangle_detail},
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=SESSION_DEFAULT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    bars_path = args.snapshot / "data/stocks/000008.csv"
    png, detail = render(args.session, bars_path, args.output)
    print(png)
    print(detail)


if __name__ == "__main__":
    main()
