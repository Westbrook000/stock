"""Load human chart labels and compare them with machine output (not returns)."""

import json
from pathlib import Path


_LABELS = {
    "W": ("w_bottom", "W底"),
    "S": ("spring", "破底翻"),
    "WS": ("w_bottom_spring", "W底破底翻"),
    "BF+": ("bull_flag", "下飘旗形上涨中继"),
    "IHS": ("inverse_head_shoulders", "头肩底"),
    "TB": ("triangle_bottom", "收敛三角形底部"),
    "M": ("m_top", "M头"),
    "FB": ("false_breakout", "假突破"),
    "FBHS": ("false_breakout_head_shoulders", "假突破（头肩顶）"),
    "BF-": ("bear_flag", "上飘旗形下跌中继"),
    "HS": ("head_shoulders", "头肩顶"),
    "TT": ("triangle_top", "收敛三角形头部"),
    "N": (None, "没有任何书本形态"),
}
_TERMINAL_STATES = {"invalidated", "target_reached", "followthrough_confirmed"}


def _has_confirmation(pattern):
    if pattern.get("state") in {"confirmed", "target_reached", "followthrough_confirmed"}:
        return True
    return any(item.get("code") in {"close_confirmation", "close_reclaim"} and item.get("value") is True
               for item in pattern.get("evidence", []))


def load_labels(path, clarifications_path=None):
    """Return normalized labels while preserving the user's formation state."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration answers must be an object keyed by case ID")
    metadata = payload.get("_meta", {})
    if not isinstance(metadata, dict):
        raise ValueError("calibration _meta must be an object")
    metadata = dict(metadata)
    if not metadata and Path(path).name == "round_01_v2.user.json":
        metadata.update({
            "calibration_round": "round_01_v2",
            "calibration_pack_version": "legacy-v1",
            "engine_version": "caisen-v14",
            "source_rule_version": "caisen-rules-v5",
        })
    labels = []
    for case_id, raw in payload.items():
        if case_id == "_meta":
            continue
        if not isinstance(raw, dict):
            raise ValueError("case %s must be an object" % case_id)
        code = raw.get("pattern")
        if code not in _LABELS:
            raise ValueError("case %s has unknown pattern label %r" % (case_id, code))
        pattern_type, label = _LABELS[code]
        state = raw.get("state") or ("none" if pattern_type is None else "confirmed")
        if state not in ("forming", "confirmed", "none"):
            raise ValueError("case %s has unsupported state %r" % (case_id, state))
        kind = raw.get("kind", "stock")
        timeframe = raw.get("timeframe", "1d")
        if kind not in ("stock", "etf"):
            raise ValueError("case %s has unsupported instrument kind %r" % (case_id, kind))
        if timeframe not in ("1d", "1w"):
            raise ValueError("case %s has unsupported timeframe %r" % (case_id, timeframe))
        display = label + "，未确认" if state == "forming" and pattern_type else label
        labels.append({
            "case_id": str(case_id),
            "code": str(raw.get("code", "")),
            "name": raw.get("name", ""),
            "kind": kind,
            "timeframe": timeframe,
            "asof": raw.get("asof"),
            "pattern_type": pattern_type,
            "state": state,
            "display_label": display,
            "note": raw.get("note", ""),
            "label_source": "user",
            "calibration_round": metadata.get("calibration_round", "legacy_unversioned"),
            "calibration_pack_version": metadata.get("calibration_pack_version"),
            "label_engine_version": metadata.get("engine_version"),
            "label_source_rule_version": metadata.get("source_rule_version"),
        })
    if clarifications_path is not None:
        clarifications = json.loads(Path(clarifications_path).read_text(encoding="utf-8"))
        by_case = {label["case_id"]: label for label in labels}
        for case_id, raw in clarifications.items():
            case_id = str(case_id)
            if case_id not in by_case or not isinstance(raw, dict):
                raise ValueError("clarification %s must refer to an existing case" % case_id)
            code = raw.get("pattern")
            if code not in _LABELS:
                raise ValueError("clarification %s has unknown pattern label %r" % (case_id, code))
            pattern_type, label = _LABELS[code]
            state = raw.get("state") or ("none" if pattern_type is None else "confirmed")
            if state not in ("forming", "confirmed", "none"):
                raise ValueError("clarification %s has unsupported state %r" % (case_id, state))
            kind = raw.get("kind", by_case[case_id]["kind"])
            timeframe = raw.get("timeframe", by_case[case_id]["timeframe"])
            if kind not in ("stock", "etf"):
                raise ValueError("clarification %s has unsupported instrument kind %r" % (case_id, kind))
            if timeframe not in ("1d", "1w"):
                raise ValueError("clarification %s has unsupported timeframe %r" % (case_id, timeframe))
            by_case[case_id].update({
                "pattern_type": pattern_type, "state": state,
                "kind": kind, "timeframe": timeframe,
                "display_label": label + "，未确认" if state == "forming" and pattern_type else label,
                "note": raw.get("note", by_case[case_id].get("note", "")),
                "label_source": "user_clarification",
                "clarified_at": raw.get("clarified_at"),
                "supersedes": raw.get("supersedes"),
            })
    return labels


def compare_labels(labels, machine_patterns):
    """Summarize label disagreements; this is not an accuracy or performance score."""
    disagreements = []
    pattern_agreement = state_agreement = state_total = 0
    for label in labels:
        case_id = label.get("case_id", "")
        matched = [p for p in machine_patterns if str(p.get("code")) == label["code"]]
        for key in ("kind", "timeframe"):
            if key in label:
                matched = [p for p in matched if p.get(key) == label[key]]
        if label.get("asof"):
            matched = [p for p in matched if not p.get("asof") or p.get("asof") == label["asof"]]
        expected = label.get("pattern_type")
        active = [p for p in matched if p.get("state") not in _TERMINAL_STATES]
        same_pattern = (any(p.get("pattern_type") == expected for p in matched)
                        if expected else not active)
        if same_pattern:
            pattern_agreement += 1
        else:
            compared = matched if expected else active
            disagreements.append({"case_id": case_id, "code": label["code"], "field": "pattern_type",
                                  "user": expected, "machine": sorted(set(p.get("pattern_type") for p in compared))})
        if expected:
            state_total += 1
            same_state = same_pattern and any(
                p.get("pattern_type") == expected and
                (_has_confirmation(p) if label["state"] == "confirmed"
                 else p.get("state") == label["state"])
                for p in matched
            )
            if same_state:
                state_agreement += 1
            elif same_pattern:
                machine_states = sorted(set(p.get("state") for p in matched if p.get("pattern_type") == expected))
                disagreements.append({"case_id": case_id, "code": label["code"], "field": "state",
                                      "user": label["state"], "machine": machine_states})
    return {
        "total": len(labels),
        "pattern_agreement": pattern_agreement,
        "state_agreement": state_agreement,
        "state_total": state_total,
        "disagreements": disagreements,
        "none_label_semantics": "截至标注日期，当前活动十招均无成立中的形态；仅已终止的历史形态不构成当前候选。旧轮退役复合形态标签仍按其记录的旧规则版本读取。",
        "state_comparison_semantics": "用户的 confirmed 表示形态曾满足价格确认条件；目标已达、后续确认或确认后失效仍按已确认比较。forming 只匹配截止日仍处于形成中的候选。",
        "interpretation": "calibration_disagreements_only",
    }
