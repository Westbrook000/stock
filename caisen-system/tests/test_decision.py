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


def test_only_three_migrated_bullish_bottom_shapes_can_create_buy_plans():
    allowed = ("w_bottom", "inverse_head_shoulders", "triangle_bottom")
    excluded = ("spring", "bull_flag", "bear_flag", "m_top", "false_breakout", "head_shoulders", "triangle_top")

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
        assert all(item["pattern_type"] != pattern_type for item in result["patterns"])


def test_one_qualified_timeframe_is_sufficient_for_buy_recommendation():
    daily = pattern("w_bottom", volume_verified=True, execution_rules_verified=True)
    weekly = pattern("triangle_bottom", volume_verified=False, execution_rules_verified=True,
                     timeframe="1w", state="forming", confirmation_date=None)
    result = synthesize_decision(
        "000001", "stock", {"1d": [daily], "1w": [weekly]},
        {"1d": 10.0, "1w": 10.0}, {"1d": "usable", "1w": "usable"},
    )
    assert result["action"] == "CONSIDER_BUY"
    assert result["selected_plan"]["timeframe"] == "1d"


def test_main_bottom_shapes_can_pass_without_volume_but_keep_all_price_risk_gates():
    for pattern_type in ("w_bottom", "inverse_head_shoulders", "triangle_bottom"):
        candidate = pattern(
            pattern_type, volume_verified=False, execution_rules_verified=True,
            evidence=[{"code":"main_structure", "value":{"role":"main","signal_eligible":True}},
                      {"code":"main_selection", "value":{"selected":True}},
                      {"code":"volume_support", "value":{"status":"unknown"}}],
        )
        result = synthesize_decision(
            "000001", "stock", {"1d": [candidate], "1w": []},
            {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
        )

        assert result["action"] == "CONSIDER_BUY", pattern_type
        assert result["selected_plan"]["pattern_id"] == candidate["pattern_id"]
        assert "volume_data_unverified" not in result["selected_plan"]["reason_codes"]
        assert result["selected_plan"]["planned_risk_pct"] <= .07
        assert result["selected_plan"]["reward_risk"] >= 2.0
        assert result["selected_plan"]["data_warnings"]


def test_main_bottom_requires_explicit_structure_signal_eligibility():
    candidate = pattern(
        "w_bottom", volume_verified=False,
        evidence=[{"code":"main_structure", "value":{"role":"main","signal_eligible":False}}],
    )
    result = synthesize_decision(
        "000001", "stock", {"1d": [candidate]},
        {"1d": 10.0}, {"1d": "usable"},
    )

    assert result["action"] == "WAIT"
    assert "main_structure_not_signal_eligible" in result["candidate_plans"][0]["reason_codes"]


def test_selected_main_structure_suppresses_nested_child_buy_plan():
    main = pattern(
        "w_bottom", pattern_id="main", volume_verified=False,
        evidence=[{"code":"main_structure", "value":{"role":"main","signal_eligible":True}},
                  {"code":"main_selection", "value":{"selected":True}}],
    )
    child = pattern("bull_flag", pattern_id="child", parent_id="main",
                    volume_verified=True, execution_rules_verified=True,
                    evidence=[{"code":"execution_suppression",
                               "value":{"suppressed":True,"reason":"nested_in_main_bottom","parent_id":"main"}}])
    result = synthesize_decision(
        "000001", "stock", {"1d": [main, child]},
        {"1d": 10.0}, {"1d": "usable"},
    )

    assert result["action"] == "CONSIDER_BUY"
    assert [plan["pattern_id"] for plan in result["candidate_plans"]] == ["main"]
    assert result["selected_plan"]["pattern_id"] == "main"


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


def test_competing_topology_waits_without_a_buy_plan():
    candidate = pattern(flags=["ambiguous_structure"])
    result = synthesize_decision(
        "000001", "stock", {"1d": [candidate], "1w": []},
        {"1d": 10.0}, {"1d": "usable", "1w": "insufficient"},
    )

    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert "ambiguous_structure" in result["reason_codes"]
    assert "多个候选形态共用结构点，暂不生成买卖计划" in result["explanation"]
    assert result["patterns"][0]["display_label"].endswith("结构存在歧义")


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
    for pattern_type in ("w_bottom", "inverse_head_shoulders", "triangle_bottom"):
        result = synthesize_decision(
            "000001", "stock", {"1d": [pattern(pattern_type, state="forming")]},
            {"1d": 10.0}, {"1d": "usable"},
        )

        assert result["action"] == "WAIT", pattern_type
        assert result["reason_codes"] == ["pattern_waiting_for_breakout"], pattern_type
        assert result["candidate_plans"] == [], pattern_type
        assert result["patterns"][0]["state"] == "forming", pattern_type
        assert "未确认" in result["patterns"][0]["display_label"], pattern_type


def test_unsupported_bull_flag_is_not_used_by_the_new_decision_path():
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
    assert result["reason_codes"] == ["no_current_pattern"]
    assert result["patterns"] == []


def test_unsupported_bearish_pattern_does_not_enter_new_path_conflict_logic():
    result = synthesize_decision(
        "000001", "stock",
        {"1d": [pattern(direction="bullish")], "1w": [pattern("head_shoulders", "bearish")]},
        {"1d": 10.0, "1w": 10.0}, {"1d": "usable", "1w": "usable"},
    )

    assert result["action"] == "WAIT"
    assert result["timeframe_conflict"] is False
    assert "timeframe_direction_conflict" not in result["reason_codes"]
    assert {item["pattern_type"] for item in result["patterns"]} == {"w_bottom"}


def test_unsupported_bearish_signal_does_not_trigger_sell_warning():
    result = synthesize_decision(
        "000001", "stock", {"1d": [pattern("m_top", "bearish")]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    assert result["action"] == "WAIT"
    assert result["reason_codes"] == ["no_current_pattern"]
    assert result["order_submitted"] is False


def test_ambiguous_bearish_topology_does_not_trigger_sell_warning():
    result = synthesize_decision(
        "000001", "stock",
        {"1d": [pattern("head_shoulders", "bearish", flags=["ambiguous_structure"])]},
        {"1d": 10.0}, {"1d": "usable"},
    )
    assert result["action"] == "WAIT"
    assert result["selected_plan"] is None
    assert result["reason_codes"] == ["no_current_pattern"]


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


def test_main_bottom_uses_frozen_neckline_stop_and_keeps_bottom_as_reference():
    candidate = pattern(
        "w_bottom", pattern_id="main-stop", invalidation=8.0, targets=[16.0],
        evidence=[
            {"code":"main_structure","value":{"role":"main","signal_eligible":True,
             "neckline_at_observed":13.0,"plan_stop_level":12.98,"support_lower":8.0,
             "support_upper":8.5,"first_trough_date":"2024-01-01"}},
            {"code":"main_selection","value":{"selected":True}},
        ],
    )
    result = synthesize_decision(
        "000001", "stock", {"1d":[candidate]},
        {"1d":13.9}, {"1d":"usable"},
    )

    plan = result["selected_plan"]
    assert plan["decision"] == "CONSIDER_BUY"
    assert plan["structure_level"] == 13.0
    assert plan["stop"] == 12.98
    assert plan["stop_basis"] == "frozen_neckline"
    assert plan["main_bottom_reference"]["support_lower"] == 8.0
    assert plan["planned_risk_pct"] <= .07
    assert plan["reward_risk"] >= 2.0
