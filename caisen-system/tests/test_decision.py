from caisen.decision import rank_scan_results, synthesize_decision


def pattern(pattern_type="w_bottom", direction="bullish", state="confirmed", targets=None, **extra):
    result = {
        "pattern_id": "id-" + pattern_type,
        "code": "000001",
        "kind": "stock",
        "timeframe": "1d",
        "pattern_type": pattern_type,
        "direction": direction,
        "state": state,
        "formed_at": "2026-10-01",
        "confirmed_at": 3,
        "confirmation_date": "2026-10-10" if state == "confirmed" else None,
        "invalidation": 9.5,
        "targets": targets if targets is not None else [12.0],
        "volume_verified": False,
        "execution_rules_verified": False,
    }
    result.update(extra)
    return result


def test_confirmed_bullish_plan_becomes_candidate_with_zero_cost_and_warnings():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern()], "1w": []},
        {"1d": 10.0}, {"1d": "usable_with_warnings", "1w": "insufficient"},
    )

    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert "volume_data_unverified" in result["candidate_plans"][0]["reason_codes"]
    assert result["data_quality"]["1w"] == "insufficient"
    assert "volume unit or corporate-action adjustment is unverified" in result["candidate_plans"][0]["data_warnings"]
    assert "研究观察" in result["explanation"]
    assert "probability" not in result


def test_only_four_approved_bullish_shapes_can_create_buy_plans():
    allowed = ("w_bottom", "inverse_head_shoulders", "bull_flag", "triangle_bottom")
    excluded = ("spring", "bear_flag", "m_top", "false_breakout", "head_shoulders", "triangle_top")

    for pattern_type in allowed:
        candidate = pattern(pattern_type, volume_verified=True, execution_rules_verified=True)
        result = synthesize_decision(
            "000001", "stock", {"1d": [candidate], "1w": []},
            {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
        )
        assert result["selected_plan"]["decision"] == "CONSIDER_BUY", pattern_type

    for pattern_type in excluded:
        candidate = pattern(pattern_type, direction="bullish", volume_verified=True,
                            execution_rules_verified=True)
        result = synthesize_decision(
            "000001", "stock", {"1d": [candidate], "1w": []},
            {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
        )
        assert result["action"] == "WAIT", pattern_type
        assert result["candidate_plans"] == [], pattern_type
        assert any(item["pattern_type"] == pattern_type for item in result["patterns"])


def test_target_reached_shape_is_visible_but_not_actionable():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern(state="target_reached")], "1w": []},
        {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
    )
    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert result["patterns"][0]["display_label"].endswith("目标已达")


def test_invalidated_bullish_shape_remains_visible_but_does_not_keep_hold_signal():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern(state="invalidated")]},
        {"1d": 10.0}, {"1d": "usable"}, has_position=True,
    )

    assert result["action"] == "WAIT"
    assert result["reason_codes"] == ["no_current_pattern"]
    assert result["patterns"][0]["display_label"].endswith("已失效")


def test_confirmed_but_competing_topology_requires_review_before_buy_plan():
    candidate = pattern(flags=["manual_review_required"])
    result = synthesize_decision(
        "000001", "stock", {"1d": [candidate], "1w": []},
        {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
    )

    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert "pattern_requires_manual_review" in result["reason_codes"]
    assert "多个候选形态共用结构点，需人工复核" in result["explanation"]
    assert result["patterns"][0]["display_label"].endswith("待人工复核")


def test_forming_pattern_waits_and_keeps_w_bottom_unconfirmed_label():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern(state="forming")]},
        {"1d": 10.0}, {"1d": "usable"},
    )

    assert result["action"] == "WAIT"
    assert result["patterns"][0]["display_label"] == "W底，未确认"
    assert "pattern_waiting_for_breakout" in result["reason_codes"]
    assert result["candidate_plans"] == []


def test_each_buy_candidate_stays_visible_and_waits_before_its_confirmation_breakout():
    for pattern_type in ("w_bottom", "inverse_head_shoulders", "bull_flag", "triangle_bottom"):
        result = synthesize_decision(
            "000001", "stock", {"1d": [pattern(pattern_type, state="forming")]},
            {"1d": 10.0}, {"1d": "usable"},
        )

        assert result["action"] == "WAIT", pattern_type
        assert result["reason_codes"] == ["pattern_waiting_for_breakout"], pattern_type
        assert result["candidate_plans"] == [], pattern_type
        assert result["patterns"][0]["state"] == "forming", pattern_type
        assert "未确认" in result["patterns"][0]["display_label"], pattern_type


def test_bull_flag_boundary_break_waits_until_prior_high_unlocks_target():
    locked_flag = pattern(
        "bull_flag", volume_verified=True, execution_rules_verified=True,
        evidence=[{"code": "measured_target_locked", "value": True}],
    )
    result = synthesize_decision(
        "000001", "stock", {"1d": [locked_flag]},
        {"1d": 10.0}, {"1d": "usable"},
    )

    assert result["action"] == "WAIT"
    assert result["candidate_plans"] == []
    assert "bull_flag_target_locked" in result["reason_codes"]


def test_confirmed_timeframe_conflict_waits_and_names_both_directions():
    result = synthesize_decision(
        "000001", "stock",
        {"1d": [pattern(direction="bullish")], "1w": [pattern("head_shoulders", "bearish")]},
        {"1d": 10.0, "1w": 10.0}, {"1d": "usable", "1w": "usable"},
    )

    assert result["action"] == "WAIT"
    assert result["timeframe_conflict"] is True
    assert "timeframe_direction_conflict" in result["reason_codes"]


def test_confirmed_bearish_signal_is_sell_if_held_not_short_order():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern("m_top", "bearish")]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    assert result["action"] == "SELL_IF_HELD"
    assert result["order_submitted"] is False


def test_ambiguous_bearish_topology_does_not_trigger_sell_warning():
    result = synthesize_decision(
        "000001", "stock",
        {"1d": [pattern("head_shoulders", "bearish", flags=["manual_review_required"])]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert "pattern_requires_manual_review" in result["reason_codes"]


def test_bullish_structure_with_existing_position_can_be_hold():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern(state="forming")]},
        {"1d": 10.0}, {"1d": "usable"}, has_position=True,
    )
    assert result["action"] == "HOLD"


def test_excessive_risk_or_insufficient_first_target_space_waits():
    too_risky = synthesize_decision(
        "000001", "stock", {"1d": [pattern(invalidation=9.0)]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    too_close = synthesize_decision(
        "000001", "stock", {"1d": [pattern(targets=[10.9])]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    assert too_risky["action"] == "WAIT"
    assert "structural_risk_over_7pct" in too_risky["reason_codes"]
    assert too_close["action"] == "WAIT"
    assert "net_first_target_below_2R" in too_close["reason_codes"]


def test_missing_weekly_data_never_counts_as_a_confirmed_timeframe():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern(volume_verified=True)], "1w": []},
        {"1d": 10.0}, {"1d": "usable_with_warnings", "1w": "insufficient"},
    )
    assert result["action"] == "CONSIDER_BUY"
    assert result["rank"]["confirmed_timeframes"] == 1


def test_missing_bars_is_insufficient_and_empty_shape_is_wait():
    missing = synthesize_decision("000001", "stock", {}, {}, {})
    empty = synthesize_decision("000001", "stock", {"1d": []}, {"1d": 10}, {"1d": "usable"})
    assert missing["action"] == "INSUFFICIENT_DATA"
    assert empty["action"] == "WAIT"


def test_buy_scan_sort_is_rule_based_and_stable():
    records = [
        {"code": "000003", "kind": "stock", "action": "WAIT", "rank": {"confirmed_timeframes": 0}},
        {"code": "000002", "kind": "stock", "action": "CONSIDER_BUY", "rank": {"confirmed_timeframes": 1, "reward_risk": 2.1, "risk_pct": .05, "confirmation_ordinal": 739900}},
        {"code": "000001", "kind": "stock", "action": "CONSIDER_BUY", "rank": {"confirmed_timeframes": 2, "reward_risk": 2.0, "risk_pct": .05, "confirmation_ordinal": 739800}},
    ]

    assert [item["code"] for item in rank_scan_results(records)] == ["000001", "000002", "000003"]
