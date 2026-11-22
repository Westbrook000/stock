from research.experiments.structure_boundary_losses_20261112.audit_paired100_e1_lifecycle_attempt02 import (
    _declared_run_counts,
    _line_equal,
    _historical_eligibility_matches_shape_evidence,
    _scope_matches_declared_cohort,
)
import json


def test_declared_run_counts_use_manifest_and_cross_check_summaries():
    manifest = {"universe": {"selected_account_count": 2}, "total": 2, "completed": 2,
                "summary": {"account_count": 2, "signal_count": 3}}
    run_summary = {"account_count": 2, "summary": {"account_count": 2, "signal_count": 3}}
    daily = {"summary": {"account_count": 2, "signal_count": 3}}

    assert _declared_run_counts(manifest, run_summary, daily) == (2, 3, True)


def test_line_equal_compares_frozen_slope_and_intercept():
    assert _line_equal({"slope": 0.1, "intercept": 2.0},
                       {"slope": 0.1, "intercept": 2.0})
    assert not _line_equal({"slope": 0.1, "intercept": 2.0},
                           {"slope": 0.2, "intercept": 2.0})


def test_main_shape_history_must_match_main_structure_and_joint_lifecycle():
    pattern = {"pattern_type": "w_bottom", "evidence": [
        {"code": "main_structure", "value": {"entry_eligible_at_confirmation": True}},
        {"code": "joint_lifecycle", "value": {
            "entry_eligible_at_confirmation": True,
            "entry_eligibility_at_confirmation_status": "known"}},
    ]}
    assert _historical_eligibility_matches_shape_evidence(pattern)


def test_flag_history_uses_flag_lifecycle_and_requires_joint_consistency():
    pattern = {"pattern_type": "bull_flag", "evidence": [
        {"code": "flag_lifecycle", "value": {
            "entry_eligible_at_confirmation": True,
            "entry_eligibility_at_confirmation_status": "known"}},
        {"code": "joint_lifecycle", "value": {
            "entry_eligible_at_confirmation": True,
            "entry_eligibility_at_confirmation_status": "known"}},
    ]}
    assert _historical_eligibility_matches_shape_evidence(pattern)
    pattern["evidence"][1]["value"]["entry_eligible_at_confirmation"] = False
    assert not _historical_eligibility_matches_shape_evidence(pattern)


def test_legacy_flag_unknown_in_joint_lifecycle_stays_missing():
    pattern = {"pattern_type": "bull_flag", "evidence": [
        {"code": "flag_lifecycle", "value": {
            "entry_eligible_at_confirmation": True,
            "entry_eligibility_at_confirmation_status": "known"}},
        {"code": "joint_lifecycle", "value": {
            "entry_eligible_at_confirmation": None,
            "entry_eligibility_at_confirmation_status": "unknown_missing_confirmation_evidence"}},
    ]}
    assert not _historical_eligibility_matches_shape_evidence(pattern)


def test_declared_run_counts_detect_summary_count_disagreement():
    manifest = {"universe": {"selected_account_count": 2}, "total": 2, "completed": 2,
                "summary": {"account_count": 2, "signal_count": 3}}
    run_summary = {"account_count": 2, "summary": {"account_count": 2, "signal_count": 3}}
    daily = {"summary": {"account_count": 2, "signal_count": 4}}

    assert _declared_run_counts(manifest, run_summary, daily) == (2, 3, False)


def test_paired100_scope_matches_fixed_member_file(tmp_path):
    members_path = tmp_path / "members.json"
    stock_codes = ["%06d" % code for code in range(100)]
    members_path.write_text(json.dumps({"stocks": stock_codes, "etfs": []}))
    manifest = {"universe": {"stock_count": 800, "etf_count": 200,
                             "selected_account_count": 100}}
    run_summary = {"sample": "paired100", "paired_members_path": str(members_path)}
    account_keys = {("stock", code) for code in stock_codes}

    assert _scope_matches_declared_cohort(manifest, run_summary, account_keys)
    assert not _scope_matches_declared_cohort(
        manifest, run_summary, account_keys - {("stock", "000000")} | {("stock", "000999")})


def test_all1000_scope_requires_800_stocks_and_200_etfs():
    manifest = {"universe": {"stock_count": 800, "etf_count": 200,
                             "selected_account_count": 1000}}
    run_summary = {"sample": "all1000"}
    accounts = ({("stock", "%06d" % code) for code in range(800)}
                | {("etf", "%06d" % code) for code in range(200)})

    assert _scope_matches_declared_cohort(manifest, run_summary, accounts)
    assert not _scope_matches_declared_cohort(
        manifest, run_summary, accounts - {("etf", "000000")} | {("stock", "000000")})
