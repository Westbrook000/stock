#!/usr/bin/env python3
"""Deep quantitative diagnosis of win-rate, return, structure, neckline, and selection drivers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from caisen.daily_research import load_research_bars  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402


def _bucket_stats(rows):
    n = len(rows)
    if not n:
        return {
            "trade_count": 0,
            "win_rate": None,
            "net_pnl": 0.0,
            "avg_return_pct": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "profit_factor": None,
            "median_holding_sessions": None,
        }
    pnls = [float(r["pnl"]) for r in rows]
    rets = [float(r["return_pct"]) * 100.0 for r in rows]
    wins = [r for r in rows if r["pnl"] > 0]
    losses = [r for r in rows if r["pnl"] < 0]
    gross_win = sum(float(r["pnl"]) for r in wins)
    gross_loss = -sum(float(r["pnl"]) for r in losses)
    holds = [int(r.get("holding_sessions") or 0) for r in rows]
    return {
        "trade_count": n,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / n, 4),
        "net_pnl": round(sum(pnls), 2),
        "avg_return_pct": round(sum(rets) / n, 2),
        "avg_win_pct": round(sum(r["return_pct"] * 100.0 for r in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(r["return_pct"] * 100.0 for r in losses) / len(losses), 2) if losses else None,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "median_holding_sessions": round(statistics.median(holds), 1) if holds else None,
    }


def _group_by(rows, key_fn, order=None):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    keys = order if order is not None else sorted(buckets.keys(), key=lambda x: str(x))
    return {str(k): _bucket_stats(buckets[k]) for k in keys if k in buckets}


def _days_between(d1, d2):
    if not d1 or not d2:
        return None
    try:
        return (date.fromisoformat(str(d2)[:10]) - date.fromisoformat(str(d1)[:10])).days
    except ValueError:
        return None


def _span_bucket(days):
    if days is None:
        return "unknown"
    if days < 30:
        return "<30d"
    if days < 90:
        return "30-89d"
    if days < 180:
        return "90-179d"
    return ">=180d"


def _slope_bucket(rel_decline, pattern_type):
    if pattern_type == "bull_flag":
        return "bull_flag"
    if rel_decline is None:
        return "unknown"
    # rel_decline = (start - end) / start; positive means downward sloping neckline
    slope_pct = -rel_decline * 100.0
    if slope_pct >= -1e-6:
        return ">=0% (horizontal/rising)"
    if slope_pct >= -5.0:
        return "-5%..0%"
    if slope_pct >= -10.0:
        return "-10%..-5%"
    if slope_pct >= -20.0:
        return "-20%..-10%"
    return "<-20%"


def _rel_height_bucket(h):
    if h is None:
        return "unknown"
    pct = h * 100.0
    if pct < 15:
        return "<15%"
    if pct < 30:
        return "15-30%"
    if pct < 60:
        return "30-60%"
    return "60-100%"


def _target_pct_bucket(entry_price, target_price):
    if not entry_price or not target_price or entry_price <= 0:
        return "unknown"
    pct = (target_price / entry_price - 1.0) * 100.0
    if pct < 15:
        return "<15%"
    if pct < 25:
        return "15-25%"
    if pct < 50:
        return "25-50%"
    return ">=50%"


def _breakout_dist_bucket(signal_close, frozen_neckline):
    if not signal_close or not frozen_neckline or frozen_neckline <= 0:
        return "unknown"
    pct = (signal_close / frozen_neckline - 1.0) * 100.0
    if pct < 2.0:
        return "<2%"
    if pct < 5.0:
        return "2-5%"
    if pct < 10.0:
        return "5-10%"
    return ">=10%"


def _gap_bucket(entry_price, signal_close):
    if not entry_price or not signal_close or signal_close <= 0:
        return "unknown"
    pct = (entry_price / signal_close - 1.0) * 100.0
    if pct < -1.0:
        return "<-1% (gap_down)"
    if pct <= 1.0:
        return "-1%..+1%"
    if pct <= 3.0:
        return "+1%..+3%"
    return ">+3% (chase_gap)"


def _vol_bucket(vr):
    if vr is None or not math.isfinite(vr):
        return "unknown"
    if vr < 1.0:
        return "<1.0x"
    if vr < 1.5:
        return "1.0-1.5x"
    if vr < 2.5:
        return "1.5-2.5x"
    return ">=2.5x"


def _shadow_bucket(sr):
    if sr is None or not math.isfinite(sr):
        return "unknown"
    if sr < 0.2:
        return "<0.20 (strong_close)"
    if sr < 0.4:
        return "0.20-0.40"
    return ">=0.40 (long_upper_wick)"


def _entry_chase_bucket(pct):
    if pct is None or not math.isfinite(pct):
        return "unknown"
    if pct < 3.0:
        return "<3%"
    if pct < 6.0:
        return "3-6%"
    if pct < 8.0:
        return "6-8%"
    if pct < 12.0:
        return "8-12%"
    return ">=12%"


def _impulse_pct_bucket(pct):
    if pct is None or not math.isfinite(pct):
        return "unknown"
    if pct < 15.0:
        return "<15%"
    if pct < 25.0:
        return "15-25%"
    if pct < 40.0:
        return "25-40%"
    return ">=40%"


def _corr_ratio_bucket(ratio):
    if ratio is None or not math.isfinite(ratio):
        return "unknown"
    if ratio < 0.30:
        return "<0.30"
    if ratio < 0.50:
        return "0.30-0.50"
    if ratio < 0.70:
        return "0.50-0.70"
    return ">=0.70"


def _extract_pattern_scale(pat):
    if not isinstance(pat, dict):
        return "unknown"
    ptype = pat.get("pattern_type")
    if ptype == "bull_flag":
        return "flag"
    ev = {e.get("code"): e.get("value") for e in (pat.get("evidence") or []) if isinstance(e, dict)}
    ms = ev.get("main_structure") or {}
    kp = pat.get("key_points") or {}
    return str(ms.get("pivot_scale") or kp.get("pivot_scale") or "unknown")


def analyze_directory(research_dir: Path):
    account_dir = research_dir / "accounts"
    found = {(item.kind, item.code): item for item in discover_instruments(ROOT)}

    records = []
    signal_competition_counts = Counter()
    rejection_reasons = Counter()
    failed_accounts = []
    trade_count_dist = Counter({"0": 0, "1-5": 0, "6-15": 0, "16-30": 0, ">30": 0})
    listing_era_counts = Counter({"<=2005": 0, "2006-2015": 0, "2016-2020": 0, ">=2021": 0, "unknown": 0})
    sessions_per_account = []
    active_account_years_total = 0.0
    measured_target_already_passed_audit = []

    scale_rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
    main_shapes = {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}

    account_paths = sorted(account_dir.glob("*.json"))
    for path in account_paths:
        acc = json.loads(path.read_text(encoding="utf-8"))
        kind = acc.get("kind")
        code = acc.get("code")
        if acc.get("status") != "complete":
            failed_accounts.append({"kind": kind, "code": code, "status": acc.get("status"), "error": acc.get("error")})
        for rej in acc.get("rejections", []):
            rejection_reasons[rej.get("reason", "unknown")] += 1

        prov = acc.get("provenance") or {}
        first_date_str = str(prov.get("first_date") or "")[:10]
        last_date_str = str(prov.get("last_date") or "")[:10]
        if first_date_str and first_date_str[:4].isdigit():
            yr = int(first_date_str[:4])
            if yr <= 2005:
                listing_era_counts["<=2005"] += 1
            elif yr <= 2015:
                listing_era_counts["2006-2015"] += 1
            elif yr <= 2020:
                listing_era_counts["2016-2020"] += 1
            else:
                listing_era_counts[">=2021"] += 1
        else:
            listing_era_counts["unknown"] += 1
        norm_rows = prov.get("normalized_rows")
        if norm_rows is not None:
            sessions_per_account.append(int(norm_rows))
        elif acc.get("daily_equity"):
            sessions_per_account.append(len(acc["daily_equity"]))
        span_d = _days_between(first_date_str, last_date_str)
        if span_d is not None and span_d > 0:
            active_account_years_total += span_d / 365.25

        signals_by_date = {}
        for sig in acc.get("signals", []):
            sdate = str(sig.get("signal_date", ""))[:10]
            signals_by_date[sdate] = sig
            pids = sig.get("pattern_ids") or []
            if len(pids) > 1:
                signal_competition_counts["multi_pattern_signal_days"] += 1
                types = sig.get("pattern_types") or []
                if types:
                    signal_competition_counts["winner_" + str(types[0])] += 1

        trades = acc.get("trades", [])
        n_trades = len(trades)
        if n_trades == 0:
            trade_count_dist["0"] += 1
            continue
        elif n_trades <= 5:
            trade_count_dist["1-5"] += 1
        elif n_trades <= 15:
            trade_count_dist["6-15"] += 1
        elif n_trades <= 30:
            trade_count_dist["16-30"] += 1
        else:
            trade_count_dist[">30"] += 1

        bars_df = None
        date_to_idx = {}
        inst = found.get((kind, code))
        if inst is not None:
            bars_df = load_research_bars(inst, root=ROOT)["bars"]
            if not bars_df.empty:
                date_str = bars_df["date"].dt.strftime("%Y-%m-%d").tolist()
                date_to_idx = {d: idx for idx, d in enumerate(date_str)}

        for trade in trades:
            sdate = str(trade.get("signal_date", ""))[:10]
            edate = str(trade.get("entry_date", ""))[:10]
            xdate = str(trade.get("exit_date", ""))[:10]
            sig = signals_by_date.get(sdate, {})
            primary_id = (trade.get("pattern_ids") or [None])[0]
            pattern_type = (trade.get("pattern_types") or ["unknown"])[0]

            # Match primary pattern snapshot
            pat_dict = None
            for cand in sig.get("patterns") or ([] if not sig.get("pattern") else [sig.get("pattern")]):
                if cand and cand.get("pattern_id") == primary_id:
                    pat_dict = cand
                    break
            if pat_dict is None and sig.get("pattern"):
                pat_dict = sig.get("pattern")

            tag = None
            for t in trade.get("pattern_tags") or []:
                if t.get("pattern_id") == primary_id:
                    tag = t
                    break
            if tag is None and trade.get("pattern_tags"):
                tag = trade["pattern_tags"][0]

            ev = {e.get("code"): e.get("value") for e in (pat_dict or {}).get("evidence", []) if isinstance(e, dict)}
            ms = ev.get("main_structure") or (tag or {}).get("main_structure") or {}
            ns = ev.get("neckline_selection") or (tag or {}).get("neckline_selection") or {}
            nb = ev.get("neckline_breakout") or {}
            fb = ev.get("flag_breakout") or {}

            pivot_scale = ms.get("pivot_scale") or ((pat_dict or {}).get("key_points") or {}).get("pivot_scale") or ("flag" if pattern_type == "bull_flag" else "unknown")
            first_trough_date = ms.get("first_trough_date") or (pat_dict or {}).get("formed_at")
            latest_anchor_date = ms.get("latest_anchor_date")
            anchor_span_days = _days_between(first_trough_date, latest_anchor_date)
            wait_span_days = _days_between(first_trough_date, sdate)

            decline_check = ns.get("decline_check") or {}
            rel_decline = decline_check.get("relative_decline")
            test_count = ms.get("neckline_independent_test_count") or len(ns.get("independent_tests") or [])
            eff_start = ns.get("effective_start_date") or ms.get("neckline_effective_start")
            remote_anchor = (bool(eff_start and first_trough_date and eff_start < first_trough_date)
                             if pattern_type != "bull_flag" else None)
            rel_height = (ns.get("neckline_relative_height") or {}).get("relative_height")
            frozen_neckline = nb.get("frozen_neckline") or fb.get("frozen_boundary")
            entry_price_val = float(trade["entry_price"])
            entry_above_neckline_pct = (
                (entry_price_val / float(frozen_neckline) - 1.0) * 100.0
                if frozen_neckline and float(frozen_neckline) > 0 else None
            )

            # Primary matching competition
            sig_patterns = sig.get("patterns") or ([] if not sig.get("pattern") else [sig.get("pattern")])
            competition_pair = None
            scale_override = "non_main_or_single"
            if len(sig_patterns) >= 2:
                winner_pat, runner_pat = sig_patterns[0], sig_patterns[1]
                w_type = str((winner_pat or {}).get("pattern_type") or "unknown")
                r_type = str((runner_pat or {}).get("pattern_type") or "unknown")
                w_scale = _extract_pattern_scale(winner_pat)
                r_scale = _extract_pattern_scale(runner_pat)
                competition_pair = f"{w_type}({w_scale}) > {r_type}({r_scale})"
                if w_type in main_shapes and r_type in main_shapes and w_scale in scale_rank and r_scale in scale_rank:
                    if scale_rank[w_scale] < scale_rank[r_scale]:
                        scale_override = "finer_scale_beat_coarser"
                    elif scale_rank[w_scale] == scale_rank[r_scale]:
                        scale_override = "same_scale"
                    else:
                        scale_override = "coarser_beat_finer"

            # Bull flag autopsy fields
            fg = ev.get("flag_geometry") or {}
            kp = (pat_dict or {}).get("key_points") or {}
            flag_impulse = fg.get("impulse") if fg.get("impulse") is not None else trade.get("impulse")
            flag_pivot1 = kp.get("pivot_1")
            flag_impulse_pct = (
                (float(flag_impulse) / float(flag_pivot1)) * 100.0
                if flag_impulse is not None and flag_pivot1 and float(flag_pivot1) > 0 else None
            )
            flag_corr_ratio = float(fg["correction_ratio"]) if fg.get("correction_ratio") is not None else None
            flag_unlock_group = None
            if pattern_type == "bull_flag":
                if trade.get("measured_target_unlocked_date") is not None:
                    flag_unlock_group = "unlocked_prior_high"
                elif trade.get("measured_target_status") == "pending_prior_high" or (
                    trade.get("measured_target_unlocked_date") is None and trade.get("exit_reason") != "take_profit"
                ):
                    flag_unlock_group = "never_unlocked_pending"
                else:
                    flag_unlock_group = "other"

            if trade.get("exit_reason") == "measured_target_already_passed":
                measured_target_already_passed_audit.append({
                    "kind": kind,
                    "code": code,
                    "entry_date": edate,
                    "exit_date": xdate,
                    "entry_price": entry_price_val,
                    "exit_price": float(trade["exit_price"]),
                    "pnl": float(trade["pnl"]),
                    "return_pct": float(trade["return_pct"]),
                    "account_status": acc.get("status"),
                    "accounting_checks": acc.get("accounting_checks"),
                    "order_events": [
                        oe for oe in acc.get("order_events", [])
                        if oe.get("date") in (edate, xdate) or str(oe.get("date", "")) >= edate
                    ][:20],
                })

            # OHLCV-derived metrics
            signal_close = nb.get("close") or fb.get("close")
            vol_ratio = None
            upper_shadow = None
            closed_below_neckline_before_or_on_stop = None
            max_single_day_drop_pct = None

            if bars_df is not None and not bars_df.empty:
                s_idx = date_to_idx.get(sdate)
                e_idx = date_to_idx.get(edate)
                x_idx = date_to_idx.get(xdate)
                if s_idx is not None:
                    row_s = bars_df.iloc[s_idx]
                    if signal_close is None:
                        signal_close = float(row_s["close"])
                    hi_s, lo_s, cl_s = float(row_s["high"]), float(row_s["low"]), float(row_s["close"])
                    if hi_s > lo_s:
                        upper_shadow = (hi_s - cl_s) / (hi_s - lo_s)
                    else:
                        upper_shadow = 0.0
                    if s_idx >= 20:
                        prev_vols = bars_df["volume"].iloc[s_idx - 20:s_idx].to_numpy(dtype=float)
                        mean_v = float(np.mean(prev_vols)) if len(prev_vols) else 0.0
                        if mean_v > 0:
                            vol_ratio = float(row_s["volume"]) / mean_v

                if e_idx is not None and x_idx is not None and x_idx >= e_idx:
                    hold_slice = bars_df.iloc[max(0, e_idx - 1):x_idx + 1]
                    closes = hold_slice["close"].to_numpy(dtype=float)
                    if len(closes) >= 2:
                        day_rets = closes[1:] / closes[:-1] - 1.0
                        max_single_day_drop_pct = float(np.min(day_rets)) * 100.0

                    if trade.get("exit_reason") == "stop_loss":
                        line = ns.get("selected_line") or ms.get("neckline") or (pat_dict or {}).get("neckline")
                        if isinstance(line, dict) and line.get("slope") is not None and line.get("intercept") is not None:
                            # Align global index using signal_date offset
                            nb_idx = nb.get("index")
                            offset = (int(nb_idx) - int(s_idx)) if (nb_idx is not None and s_idx is not None) else 0
                            slope, intercept = float(line["slope"]), float(line["intercept"])
                            closed_below = False
                            for j in range(e_idx, x_idx + 1):
                                g_j = j + offset
                                line_val = slope * g_j + intercept
                                if float(bars_df.iloc[j]["close"]) < line_val:
                                    closed_below = True
                                    break
                            closed_below_neckline_before_or_on_stop = closed_below

            records.append({
                "kind": kind,
                "code": code,
                "signal_date": sdate,
                "entry_date": edate,
                "entry_year": edate[:4] if edate else "unknown",
                "exit_date": xdate,
                "entry_price": float(trade["entry_price"]),
                "exit_price": float(trade["exit_price"]),
                "target_price": trade.get("target_price"),
                "stop_price": trade.get("stop_price"),
                "pnl": float(trade["pnl"]),
                "return_pct": float(trade["return_pct"]),
                "holding_sessions": int(trade.get("holding_sessions") or 0),
                "exit_reason": trade.get("exit_reason"),
                "pattern_type": pattern_type,
                "pivot_scale": pivot_scale,
                "first_trough_date": first_trough_date,
                "latest_anchor_date": latest_anchor_date,
                "anchor_span_days": anchor_span_days,
                "wait_span_days": wait_span_days,
                "rel_decline": rel_decline,
                "slope_bucket": _slope_bucket(rel_decline, pattern_type),
                "test_count": int(test_count) if pattern_type != "bull_flag" else None,
                "remote_anchor": remote_anchor,
                "rel_height": rel_height,
                "frozen_neckline": frozen_neckline,
                "signal_close": signal_close,
                "vol_ratio": vol_ratio,
                "upper_shadow": upper_shadow,
                "multi_candidate_signal": len(sig.get("pattern_ids") or []) > 1,
                "competition_pair": competition_pair,
                "scale_override": scale_override,
                "entry_above_neckline_pct": entry_above_neckline_pct,
                "flag_unlock_group": flag_unlock_group,
                "flag_impulse_pct": flag_impulse_pct,
                "flag_corr_ratio": flag_corr_ratio,
                "closed_below_neckline_on_stop": closed_below_neckline_before_or_on_stop,
                "max_single_day_drop_pct": max_single_day_drop_pct,
            })

    stop_losses = [r for r in records if r["exit_reason"] == "stop_loss"]
    whipsaw_chases = [
        r["entry_above_neckline_pct"] for r in stop_losses
        if r["closed_below_neckline_on_stop"] is False and r["entry_above_neckline_pct"] is not None
    ]
    closed_below_chases = [
        r["entry_above_neckline_pct"] for r in stop_losses
        if r["closed_below_neckline_on_stop"] is True and r["entry_above_neckline_pct"] is not None
    ]
    stop_autopsy = {
        "total_stop_losses": len(stop_losses),
        "closed_below_neckline": sum(1 for r in stop_losses if r["closed_below_neckline_on_stop"] is True),
        "whipsawed_above_neckline": sum(1 for r in stop_losses if r["closed_below_neckline_on_stop"] is False),
        "unknown_neckline_stop": sum(1 for r in stop_losses if r["closed_below_neckline_on_stop"] is None),
        "whipsaw_median_entry_above_neckline_pct": round(statistics.median(whipsaw_chases), 2) if whipsaw_chases else None,
        "whipsaw_pct_entry_gt_8pct_above_neckline": (
            round(sum(1 for x in whipsaw_chases if x > 8.0) / len(whipsaw_chases), 4) if whipsaw_chases else None
        ),
        "closed_below_median_entry_above_neckline_pct": round(statistics.median(closed_below_chases), 2) if closed_below_chases else None,
        "single_day_drop_gt_15pct_count": sum(
            1 for r in records if r["max_single_day_drop_pct"] is not None and r["max_single_day_drop_pct"] < -15.0
        ),
        "single_day_drop_gt_15pct_examples": [
            {k: r[k] for k in ("kind", "code", "entry_date", "exit_date", "pattern_type", "return_pct", "max_single_day_drop_pct")}
            for r in sorted(
                (r for r in records if r["max_single_day_drop_pct"] is not None and r["max_single_day_drop_pct"] < -15.0),
                key=lambda x: x["max_single_day_drop_pct"],
            )[:20]
        ],
    }

    non_flag_records = [r for r in records if r["pattern_type"] != "bull_flag"]
    flag_records = [r for r in records if r["pattern_type"] == "bull_flag"]
    multi_records = [r for r in records if r["competition_pair"] is not None]
    pair_buckets = defaultdict(list)
    for r in multi_records:
        pair_buckets[r["competition_pair"]].append(r)
    top_pairs = sorted(pair_buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]
    by_competition_pair = {k: _bucket_stats(v) for k, v in top_pairs}

    total_net_pnl = sum(float(r["pnl"]) for r in records)
    summary_path = research_dir / "daily-research.json"
    summary_data = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8")).get("summary") or {}
        except Exception:
            summary_data = {}
    starting_capital = float(summary_data.get("starting_capital") or (len(account_paths) * 100000.0) or 100000000.0)
    reported_net_pnl = float(summary_data.get("net_pnl") if summary_data.get("net_pnl") is not None else total_net_pnl)
    cap_util = summary_data.get("capital_utilization")
    ann_years = summary_data.get("annualization_period_years")
    implied_rodc = (
        round(reported_net_pnl / (starting_capital * float(cap_util) * float(ann_years)), 4)
        if cap_util and ann_years and float(cap_util) > 0 and float(ann_years) > 0 else None
    )
    return_per_active_account_year = (
        round(reported_net_pnl / (active_account_years_total * 100000.0), 4)
        if active_account_years_total > 0 else None
    )
    capital_utilization_breakdown = {
        "trade_count_distribution": {k: trade_count_dist[k] for k in ("0", "1-5", "6-15", "16-30", ">30")},
        "listing_era_counts": {k: listing_era_counts[k] for k in ("<=2005", "2006-2015", "2016-2020", ">=2021", "unknown")},
        "avg_sessions_per_account": round(sum(sessions_per_account) / len(sessions_per_account), 1) if sessions_per_account else None,
        "median_sessions_per_account": round(statistics.median(sessions_per_account), 1) if sessions_per_account else None,
        "active_account_years_total": round(active_account_years_total, 2),
        "reported_capital_utilization": round(float(cap_util), 6) if cap_util is not None else None,
        "reported_annualization_period_years": round(float(ann_years), 2) if ann_years is not None else None,
        "implied_return_on_deployed_capital_annualized": implied_rodc,
        "return_per_active_account_year": return_per_active_account_year,
    }

    diagnosis = {
        "accounts_scanned": len(account_paths),
        "failed_accounts": failed_accounts,
        "rejection_reasons": dict(rejection_reasons),
        "overall": _bucket_stats(records),
        "by_kind": _group_by(records, lambda r: r["kind"], ["stock", "etf"]),
        "by_shape": _group_by(records, lambda r: r["pattern_type"], ["w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"]),
        "by_exit_reason": _group_by(records, lambda r: r["exit_reason"]),
        "by_pivot_scale": _group_by(records, lambda r: r["pivot_scale"], ["2atr_5pct", "3atr_8pct", "5atr_12pct", "flag", "unknown"]),
        "by_anchor_span": _group_by(records, lambda r: _span_bucket(r["anchor_span_days"]), ["<30d", "30-89d", "90-179d", ">=180d", "unknown"]),
        "by_anchor_span_non_flag": _group_by(
            non_flag_records,
            lambda r: _span_bucket(r["anchor_span_days"]),
            ["<30d", "30-89d", "90-179d", ">=180d", "unknown"],
        ),
        "by_wait_span_non_flag": _group_by(
            non_flag_records,
            lambda r: _span_bucket(r["wait_span_days"]),
            ["<30d", "30-89d", "90-179d", ">=180d", "unknown"],
        ),
        "by_neckline_slope": _group_by(
            records,
            lambda r: r["slope_bucket"],
            [">=0% (horizontal/rising)", "-5%..0%", "-10%..-5%", "-20%..-10%", "<-20%", "bull_flag", "unknown"],
        ),
        "by_test_count": _group_by(
            non_flag_records,
            lambda r: ">=4" if (r["test_count"] or 0) >= 4 else str(r["test_count"]),
            ["2", "3", ">=4"],
        ),
        "by_remote_anchor": _group_by(
            non_flag_records,
            lambda r: "pre_structure_anchor" if r["remote_anchor"] is True else ("in_or_post_structure" if r["remote_anchor"] is False else "unknown"),
        ),
        "by_relative_height": _group_by(
            non_flag_records,
            lambda r: _rel_height_bucket(r["rel_height"]),
            ["<15%", "15-30%", "30-60%", "60-100%", "unknown"],
        ),
        "by_target_pct": _group_by(
            records,
            lambda r: _target_pct_bucket(r["entry_price"], r["target_price"]),
            ["<15%", "15-25%", "25-50%", ">=50%", "unknown"],
        ),
        "by_breakout_distance": _group_by(
            records,
            lambda r: _breakout_dist_bucket(r["signal_close"], r["frozen_neckline"]),
            ["<2%", "2-5%", "5-10%", ">=10%", "unknown"],
        ),
        "by_entry_above_neckline": _group_by(
            records,
            lambda r: _entry_chase_bucket(r["entry_above_neckline_pct"]),
            ["<3%", "3-6%", "6-8%", "8-12%", ">=12%", "unknown"],
        ),
        "by_open_chase_gap": _group_by(
            records,
            lambda r: _gap_bucket(r["entry_price"], r["signal_close"]),
            ["<-1% (gap_down)", "-1%..+1%", "+1%..+3%", ">+3% (chase_gap)", "unknown"],
        ),
        "by_volume_ratio": _group_by(
            records,
            lambda r: _vol_bucket(r["vol_ratio"]),
            ["<1.0x", "1.0-1.5x", "1.5-2.5x", ">=2.5x", "unknown"],
        ),
        "by_upper_shadow": _group_by(
            records,
            lambda r: _shadow_bucket(r["upper_shadow"]),
            ["<0.20 (strong_close)", "0.20-0.40", ">=0.40 (long_upper_wick)", "unknown"],
        ),
        "by_multi_candidate": _group_by(records, lambda r: "multi_candidate" if r["multi_candidate_signal"] else "single_candidate"),
        "by_scale_override": _group_by(
            records,
            lambda r: r["scale_override"],
            ["finer_scale_beat_coarser", "same_scale", "coarser_beat_finer", "non_main_or_single"],
        ),
        "by_competition_pair": by_competition_pair,
        "by_shape_x_pivot_scale": _group_by(
            non_flag_records,
            lambda r: f"{r['pattern_type']} | {r['pivot_scale']}",
        ),
        "by_shape_x_slope": _group_by(
            non_flag_records,
            lambda r: f"{r['pattern_type']} | {r['slope_bucket']}",
        ),
        "by_shape_x_remote_anchor": _group_by(
            [r for r in non_flag_records if r["remote_anchor"] is not None],
            lambda r: f"{r['pattern_type']} | {'pre_structure' if r['remote_anchor'] else 'in_structure'}",
        ),
        "by_shape_x_anchor_span": _group_by(
            non_flag_records,
            lambda r: f"{r['pattern_type']} | {_span_bucket(r['anchor_span_days'])}",
        ),
        "bull_flag_by_unlock": _group_by(
            flag_records,
            lambda r: r["flag_unlock_group"] or "unknown",
            ["unlocked_prior_high", "never_unlocked_pending", "other", "unknown"],
        ),
        "bull_flag_by_impulse_pct": _group_by(
            flag_records,
            lambda r: _impulse_pct_bucket(r["flag_impulse_pct"]),
            ["<15%", "15-25%", "25-40%", ">=40%", "unknown"],
        ),
        "bull_flag_by_correction_ratio": _group_by(
            flag_records,
            lambda r: _corr_ratio_bucket(r["flag_corr_ratio"]),
            ["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70", "unknown"],
        ),
        "signal_competition_counts": dict(signal_competition_counts),
        "by_entry_year": _group_by(records, lambda r: r["entry_year"]),
        "stop_loss_autopsy": stop_autopsy,
        "capital_utilization_breakdown": capital_utilization_breakdown,
        "measured_target_already_passed_audit": measured_target_already_passed_audit,
        "worst_15_trades": sorted(records, key=lambda r: r["return_pct"])[:15],
        "best_15_trades": sorted(records, key=lambda r: r["return_pct"], reverse=True)[:15],
    }
    return diagnosis


def _md_table(title, mapping):
    lines = [f"### {title}", "", "| 分组 | 交易数 | 胜率 | 净盈亏 (¥) | 均收益% | 均盈利% | 均亏损% | 盈亏比 PF | 中位持有日 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, st in mapping.items():
        wr = f"{st['win_rate']*100:.2f}%" if st["win_rate"] is not None else "—"
        avg_r = f"{st['avg_return_pct']:.2f}%" if st["avg_return_pct"] is not None else "—"
        avg_w = f"{st['avg_win_pct']:.2f}%" if st["avg_win_pct"] is not None else "—"
        avg_l = f"{st['avg_loss_pct']:.2f}%" if st["avg_loss_pct"] is not None else "—"
        pf = f"{st['profit_factor']:.3f}" if st["profit_factor"] is not None else "—"
        hold = f"{st['median_holding_sessions']}" if st["median_holding_sessions"] is not None else "—"
        lines.append(f"| {key} | {st['trade_count']} | {wr} | {st['net_pnl']:,.0f} | {avg_r} | {avg_w} | {avg_l} | {pf} | {hold} |")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(diag):
    sections = ["# 1,000 标的全历史回测深度诊断报告", ""]
    o = diag["overall"]
    sa = diag["stop_loss_autopsy"]
    cu = diag.get("capital_utilization_breakdown") or {}
    sections.append(
        f"- **扫描账户数**: {diag['accounts_scanned']}（异常账户 {len(diag['failed_accounts'])}）\n"
        f"- **总成交数**: {o['trade_count']} · **胜率**: {o['win_rate']*100:.2f}% · **净盈亏**: ¥{o['net_pnl']:,.2f} · **盈亏比 PF**: {o['profit_factor']}\n"
        f"- **拒单统计**: {json.dumps(diag['rejection_reasons'], ensure_ascii=False)}\n"
        f"- **同日多候选竞争**: {json.dumps(diag['signal_competition_counts'], ensure_ascii=False)}\n"
        f"- **止损解剖**: 共 {sa['total_stop_losses']} 次止损；其中 **{sa['whipsawed_above_neckline']} 次未收盘跌破冻结颈线即被 -8% 洗出**（入场高出颈线中位数 {sa.get('whipsaw_median_entry_above_neckline_pct')}%，>8%占比 {sa.get('whipsaw_pct_entry_gt_8pct_above_neckline')}），{sa['closed_below_neckline']} 次在收盘跌破颈线后止损（入场高出颈线中位数 {sa.get('closed_below_median_entry_above_neckline_pct')}%）；单日暴跌 >15% 交易 {sa['single_day_drop_gt_15pct_count']} 笔。\n"
        f"- **资金利用率与生命周期拆解**: {json.dumps(cu, ensure_ascii=False)}\n"
        f"- **measured_target_already_passed 审计数**: {len(diag.get('measured_target_already_passed_audit') or [])}\n"
    )
    for title, key in (
        ("按标的类型", "by_kind"),
        ("按形态类别", "by_shape"),
        ("按退出原因", "by_exit_reason"),
        ("按结构点尺度 (Pivot Scale)", "by_pivot_scale"),
        ("按核心锚点跨度 (非旗形 last_anchor - first_trough)", "by_anchor_span_non_flag"),
        ("按整体等待跨度 (非旗形 signal - first_trough)", "by_wait_span_non_flag"),
        ("按颈线斜率 / 累计跌幅", "by_neckline_slope"),
        ("按独立测试次数", "by_test_count"),
        ("按颈线测试起点（是否借用结构前远程高点）", "by_remote_anchor"),
        ("按颈线相对高度", "by_relative_height"),
        ("按第一测幅目标涨幅空间", "by_target_pct"),
        ("按突破收盘高出冻结颈线幅度", "by_breakout_distance"),
        ("按实际入场高出冻结颈线幅度 (追高距离)", "by_entry_above_neckline"),
        ("按次日开盘追高缺口", "by_open_chase_gap"),
        ("按突破日量能比 (Vol / MA20)", "by_volume_ratio"),
        ("按突破日上影线比例", "by_upper_shadow"),
        ("按同日多形态竞争 vs 单候选", "by_multi_candidate"),
        ("按尺度覆盖关系 (细尺度压制粗尺度 vs 反之)", "by_scale_override"),
        ("按同日多形态竞争配对 Top20 (Winner > Runner-up)", "by_competition_pair"),
        ("交叉分组: 形态 × 结构点尺度", "by_shape_x_pivot_scale"),
        ("交叉分组: 形态 × 颈线斜率", "by_shape_x_slope"),
        ("交叉分组: 形态 × 远程锚点", "by_shape_x_remote_anchor"),
        ("交叉分组: 形态 × 核心锚点跨度", "by_shape_x_anchor_span"),
        ("下飘旗形专项: 是否解锁前高测幅", "bull_flag_by_unlock"),
        ("下飘旗形专项: 第一波推动幅度 (impulse / pivot_1)", "bull_flag_by_impulse_pct"),
        ("下飘旗形专项: 旗面回撤比例 (correction_ratio)", "bull_flag_by_correction_ratio"),
        ("按入场年份", "by_entry_year"),
    ):
        sections.append(_md_table(title, diag[key]))
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("research_dir", type=Path)
    args = parser.parse_args()
    diag = analyze_directory(args.research_dir)
    out_json = args.research_dir / "diagnosis.json"
    out_md = args.research_dir / "diagnosis.md"
    out_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(diag), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")
    print(json.dumps({"overall": diag["overall"], "by_shape": diag["by_shape"], "stop_loss_autopsy": {k: v for k, v in diag["stop_loss_autopsy"].items() if k != "single_day_drop_gt_15pct_examples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
