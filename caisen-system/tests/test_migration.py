from caisen.migration import migration_status


def test_migration_status_covers_every_catalog_pattern_and_marks_legacy_families_unsupported():
    status = migration_status()
    assert status["supported_new_path"] == [
        "w_bottom", "inverse_head_shoulders", "triangle_bottom"
    ]
    assert set(status["families"]) == {
        "w_bottom", "spring", "bull_flag", "bear_flag", "m_top",
        "false_breakout", "head_shoulders", "inverse_head_shoulders",
        "triangle_top", "triangle_bottom",
    }
    assert {name for name, entry in status["families"].items()
            if entry["status"] == "unsupported"} == {
        "spring", "bull_flag", "bear_flag", "m_top", "false_breakout",
        "head_shoulders", "triangle_top",
    }
    assert all(entry["reason_code"] == "not_migrated_to_causal_main_engine"
               for entry in status["families"].values() if entry["status"] == "unsupported")
