"""Projection shared by online scans and the offline scan-history migration."""
import base64
import math

_TERMINAL = {"invalidated", "expired", "target_reached", "followthrough_confirmed"}
_PATTERN_FIELDS = ("pattern_type", "timeframe", "state", "formed_at", "confirmation_date")


def representative_patterns(row):
    plan = row.get("selected_plan") or {}
    watch = row.get("watch") or []
    relevant = ({item.get("pattern_id") for item in watch} if row.get("action") == "WATCH"
                else {plan.get("pattern_id")} if row.get("action") == "CONSIDER_BUY" else set())
    relevant.discard(None)
    for timeframe in ("1d", "1w"):
        matches = [p for p in row.get("patterns", []) or []
                   if p.get("timeframe") == timeframe and p.get("state") not in _TERMINAL]
        marked = next((p for p in matches if p.get("pattern_id") in relevant), None)
        primary = [p for p in matches if not p.get("parent_id") and not p.get("execution_suppressed")]
        main = [p for p in primary if p.get("selected_main") or p.get("is_main_structure")]
        selected = marked or max(main or primary, default=None,
                                  key=lambda p: p.get("confirmation_date") or p.get("formed_at") or "")
        if selected:
            yield {k:selected[k] for k in _PATTERN_FIELDS if selected.get(k) is not None}


def encode_mini_bars(bars):
    """Four unsigned bytes per candle: open, close, low, high, scaled to 0–255."""
    try:
        prices = [[float(bar[key]) for key in ("open", "close", "low", "high")]
                  for bar in bars[-24:]]
    except (KeyError, TypeError, ValueError):
        return None
    if not prices or not all(math.isfinite(price) for bar in prices for price in bar):
        return None
    lo = min(min(bar) for bar in prices)
    hi = max(max(bar) for bar in prices)
    encoded = bytes(round(255 * ((price - lo) / (hi - lo))) if hi > lo else 127
                    for bar in prices for price in bar)
    return base64.b64encode(encoded).decode("ascii")


def scan_summary(row):
    data = {"summary_version": 1}
    for key in ("code", "kind", "name", "category", "action", "latest_date", "latest_close", "state"):
        value = row.get(key)
        if value is not None and (not isinstance(value, float) or math.isfinite(value)):
            data[key] = value
    reasons = row.get("reason_codes") or []
    if reasons:
        data["reason_codes"] = [str(reasons[0])[:120]]
    if row.get("error"):
        data["error"] = str(row["error"])[:240]
    data["patterns"] = list(representative_patterns(row))
    plan = row.get("selected_plan") or {}
    if plan:
        data["selected_plan"] = {key:plan[key] for key in (
            "timeframe", "planned_risk_pct", "measured_target_status", "measured_target_price",
            "reward_risk") if plan.get(key) is not None
            and (not isinstance(plan[key], float) or math.isfinite(plan[key]))}
        target = next(iter(plan.get("targets") or []), None)
        if isinstance(target, (int, float)) and math.isfinite(target):
            data["selected_plan"]["targets"] = [target]
    if row.get("action") == "WATCH" and row.get("watch"):
        item = row["watch"][0]
        data["watch"] = [{key:item[key] for key in (
            "timeframe", "pattern_type", "observed_date", "boundary", "distance_atr") if item.get(key) is not None
            and (not isinstance(item[key], float) or math.isfinite(item[key]))}]
    encoded = (row.get("mini_candles_u8") if row.get("summary_version") == 1
               else encode_mini_bars(row.get("mini_bars") or []))
    if encoded:
        data["mini_candles_u8"] = encoded
    return data
