"""Explainable action labels and deterministic candidate ranking."""

from datetime import date

from .patterns import BOOK_PATTERNS
from .plans import build_trade_plan


_ACTION_ORDER = {"CONSIDER_BUY": 0, "WAIT": 1, "HOLD": 2, "SELL_IF_HELD": 3, "INSUFFICIENT_DATA": 4}


def _value(pattern, field, default=None):
    if isinstance(pattern, dict):
        return pattern.get(field, default)
    return getattr(pattern, field, default)


def _to_dict(pattern):
    return pattern.to_dict() if hasattr(pattern, "to_dict") else dict(pattern)


def _label(pattern):
    name = BOOK_PATTERNS.get(_value(pattern, "pattern_type"), _value(pattern, "pattern_type", "形态"))
    state = _value(pattern, "state")
    if state == "forming":
        label = name + "，未确认"
    elif state == "confirmed":
        label = name + ("，价格确认" if _value(pattern, "volume_verified", False) else "，价格确认，量能未核验")
    elif state == "invalidated":
        label = name + "，已失效"
    elif state == "target_reached":
        label = name + "，目标已达"
    elif state == "followthrough_confirmed":
        label = name + "，箱体后续边界已确认（历史）"
    else:
        label = name + "，" + str(state)
    if "manual_review_required" in (_value(pattern, "flags", []) or []):
        label += "，待人工复核"
    return label


def synthesize_decision(code, kind, patterns_by_timeframe, prices_by_timeframe,
                        quality_by_timeframe, bars_by_timeframe=None, cash=100000,
                        has_position=False, warnings_by_timeframe=None):
    """Synthesize one ticker row from independent daily/weekly evidence.

    ``quality_by_timeframe`` uses market data quality statuses. Missing or
    insufficient weekly data is kept as unknown and cannot create agreement.
    """
    bars_by_timeframe = bars_by_timeframe or {}
    warnings_by_timeframe = warnings_by_timeframe or {}
    timeframes = ("1d", "1w")
    patterns = [p for tf in timeframes for p in patterns_by_timeframe.get(tf, [])]
    active_patterns = [p for p in patterns if _value(p, "state") not in
                       ("invalidated", "target_reached", "followthrough_confirmed")]
    qualities = {tf: quality_by_timeframe.get(tf, "unavailable") for tf in timeframes}
    usable = any(qualities[tf] in ("usable", "usable_with_warnings") and prices_by_timeframe.get(tf) is not None for tf in timeframes)
    if not usable:
        result = _result(code, kind, "INSUFFICIENT_DATA", ["no_usable_timeframe_data"], patterns, qualities, None, 0)
        result["data_warnings"] = ["%s: %s" % (tf, warning) for tf in timeframes for warning in warnings_by_timeframe.get(tf, [])]
        return result

    confirmed = {tf: [p for p in patterns_by_timeframe.get(tf, [])
                      if _value(p, "state") == "confirmed" and
                      "manual_review_required" not in (_value(p, "flags", []) or [])]
                 for tf in timeframes}
    directions = {tf: {str(_value(p, "direction")) for p in confirmed[tf]} for tf in timeframes}
    bullish_present = any("bullish" in directions[tf] for tf in timeframes)
    bearish_present = any("bearish" in directions[tf] for tf in timeframes)
    conflict = bullish_present and bearish_present
    confirmed_timeframes = sum(bool(directions[tf]) for tf in timeframes)

    plans = []
    for tf in timeframes:
        price = prices_by_timeframe.get(tf)
        if price is None:
            continue
        for pattern in patterns_by_timeframe.get(tf, []):
            if _value(pattern, "direction") != "bullish" or _value(pattern, "state") != "confirmed":
                continue
            if "manual_review_required" in (_value(pattern, "flags", []) or []):
                continue
            plan = build_trade_plan(
                pattern, float(price), cash=cash, bars=bars_by_timeframe.get(tf),
            )
            plan["timeframe"] = tf
            plans.append(plan)

    passing = [plan for plan in plans if plan["decision"] == "CONSIDER_BUY"]
    passing.sort(key=lambda plan: (
        -(min(float(plan.get("reward_risk") or 0), 10.0)),
        float(plan.get("planned_risk_pct") or float("inf")),
        plan.get("pattern_id") or "",
        plan["timeframe"],
    ))
    selected = passing[0] if passing else None
    if conflict:
        action, reasons = "WAIT", ["timeframe_direction_conflict"]
    elif selected is not None:
        action, reasons = "CONSIDER_BUY", []
    elif bearish_present:
        action, reasons = "SELL_IF_HELD", ["confirmed_bearish_structure_sell_warning_if_held"]
    elif has_position and any(_value(p, "direction") == "bullish" and
                              "manual_review_required" not in (_value(p, "flags", []) or [])
                              for p in active_patterns):
        action, reasons = "HOLD", ["bullish_structure_exists_without_new_entry"]
    elif plans:
        action = "WAIT"
        reasons = sorted({reason for plan in plans for reason in plan.get("reason_codes", [])}) or ["buy_gates_not_met"]
    elif active_patterns:
        manual_review = any("manual_review_required" in (_value(p, "flags", []) or [])
                            for p in active_patterns)
        action, reasons = "WAIT", ["pattern_requires_manual_review" if manual_review else "no_confirmed_actionable_pattern"]
    else:
        action, reasons = "WAIT", ["no_current_pattern"]

    same_direction_confirmed = 0
    for tf in timeframes:
        if directions[tf] and len(directions[tf]) == 1 and next(iter(directions[tf])) == ("bullish" if selected else next(iter(directions[tf]))):
            same_direction_confirmed += 1
    selected_rr = float(selected.get("reward_risk") or 0) if selected else 0.0
    selected_risk = float(selected.get("planned_risk_pct") or 1.0) if selected else 1.0
    confirmation_dates = [
        str(_value(p, "confirmation_date", ""))
        for p in patterns if _value(p, "state") == "confirmed" and _value(p, "confirmation_date")
    ]
    latest_confirmation = max(confirmation_dates, default="")
    try:
        confirmation_ordinal = date.fromisoformat(latest_confirmation).toordinal()
    except ValueError:
        confirmation_ordinal = 0
    rank = {
        "action_priority": _ACTION_ORDER[action],
        "confirmed_timeframes": same_direction_confirmed,
        "reward_risk": min(selected_rr, 10.0) if selected else 0.0,
        "risk_pct": selected_risk,
        "confirmation_date": latest_confirmation or None,
        "confirmation_ordinal": confirmation_ordinal,
        "tie_break": {"kind": kind, "code": code},
    }
    quality_warnings = []
    for tf, quality in qualities.items():
        if quality not in ("usable", "usable_with_warnings"):
            quality_warnings.append("%s data status: %s" % (tf, quality))
        quality_warnings.extend("%s: %s" % (tf, warning) for warning in warnings_by_timeframe.get(tf, []))
    quality_warnings = list(dict.fromkeys(quality_warnings))
    current_patterns = []
    for pattern in patterns:
        item = _to_dict(pattern)
        item["display_label"] = _label(pattern)
        current_patterns.append(item)
    result = {
        "code": code, "kind": kind, "action": action, "reason_codes": reasons,
        "explanation": _explain(action, reasons), "patterns": current_patterns,
        "data_quality": qualities, "data_warnings": quality_warnings,
        "timeframe_conflict": conflict, "confirmed_timeframes": confirmed_timeframes,
        "selected_plan": selected, "candidate_plans": plans,
        "rank": rank, "order_submitted": False,
    }
    return result


def _result(code, kind, action, reasons, patterns, qualities, selected, confirmed_timeframes):
    return {
        "code": code, "kind": kind, "action": action, "reason_codes": reasons,
        "explanation": _explain(action, reasons),
        "patterns": [{**_to_dict(p), "display_label": _label(p)} for p in patterns],
        "data_quality": qualities, "data_warnings": [], "timeframe_conflict": False,
        "confirmed_timeframes": confirmed_timeframes, "selected_plan": selected,
        "candidate_plans": [], "rank": {"action_priority": _ACTION_ORDER[action], "confirmed_timeframes": 0,
        "reward_risk": 0, "risk_pct": 1, "confirmation_date": None, "confirmation_ordinal": 0,
        "tie_break": {"kind": kind, "code": code}}, "order_submitted": False,
    }


def _explain(action, reasons):
    if action == "CONSIDER_BUY":
        return "结构已价格确认，固定止损风险和首个有效目标空间通过零费用基线；仅供人工复核。"
    if action == "SELL_IF_HELD":
        return "检测到已确认的空头/顶部结构；若已有持仓，作为退出风险提示，不代表做空或自动卖出。"
    if action == "INSUFFICIENT_DATA":
        return "没有可用于正式判断的有效行情数据。"
    labels = {
        "timeframe_direction_conflict": "日线与周线确认方向冲突，请分别复核。",
        "pattern_requires_manual_review": "多个候选形态共用结构点，需人工复核。",
        "no_confirmed_actionable_pattern": "当前没有通过确认与交易空间检查的形态。",
        "no_current_pattern": "当前周期没有识别到有效形态。",
        "confirmed_bearish_structure_sell_warning_if_held": "已确认顶部或空头结构；如有持仓，请复核退出计划。",
        "bullish_structure_exists_without_new_entry": "存在多头结构，但当前没有新的入场信号。",
        "buy_gates_not_met": "买入条件未全部满足。",
        "pattern_not_confirmed_bullish": "形态尚未确认为多头结构。",
        "volume_data_unverified": "成交量单位或企业行动未核验，形态仅作研究观察。",
        "invalid_structural_stop": "无法建立有效的结构止损。",
        "no_verified_target": "没有可用于计划的目标位。",
        "structural_risk_over_7pct": "结构止损对应风险超过 7%。",
        "first_target_already_reached_or_below_entry": "首个目标已触及或低于当前参考价。",
        "entry_above_maximum_allowed_price": "当前参考价高于最高允许买入价。",
        "insufficient_cash_or_liquidity": "现金或流动性不足以形成有效委托。",
        "net_first_target_below_2R": "首个目标空间不足 2R。",
    }
    return "；".join(labels.get(reason, "需复核：" + reason) for reason in reasons) or action


def rank_scan_results(results):
    """Sort scan rows using disclosed keys; no score is interpreted as probability."""
    return sorted(results, key=lambda item: (
        int(item.get("rank", {}).get("action_priority", _ACTION_ORDER.get(item.get("action"), 99))),
        -int(item.get("rank", {}).get("confirmed_timeframes", 0)),
        -float(item.get("rank", {}).get("reward_risk", 0)),
        float(item.get("rank", {}).get("risk_pct", 1)),
        -int(item.get("rank", {}).get("confirmation_ordinal", 0)),
        str(item.get("kind", "")), str(item.get("code", "")),
    ))
