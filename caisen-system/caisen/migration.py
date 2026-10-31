"""Visible migration boundary for the causal main-structure analysis path."""

from .rules.catalog import load_rule_catalog


SUPPORTED_NEW_PATH_TYPES = (
    "w_bottom",
    "inverse_head_shoulders",
    "triangle_bottom",
)
UNSUPPORTED_REASON = "not_migrated_to_causal_main_engine"


def migration_status():
    """Describe support for every catalog family without hiding legacy rules."""
    supported = set(SUPPORTED_NEW_PATH_TYPES)
    families = {}
    for rule in load_rule_catalog()["patterns"]:
        pattern_type = rule["rule_id"]
        is_supported = pattern_type in supported
        families[pattern_type] = {
            "status": "supported" if is_supported else "unsupported",
            "label": rule["name"],
            "reason_code": None if is_supported else UNSUPPORTED_REASON,
        }
    return {
        "engine": "causal_main_structure_v1",
        "supported_new_path": list(SUPPORTED_NEW_PATH_TYPES),
        "families": families,
    }
