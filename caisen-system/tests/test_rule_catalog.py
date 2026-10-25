import json
from copy import deepcopy
import re
from pathlib import Path

import pytest

from caisen.patterns import BOOK_PATTERNS
from caisen.rules.catalog import RuleCatalogError, load_rule_catalog, validate_rule_catalog


def test_catalog_covers_all_book_pattern_ids_with_verified_sources():
    catalog = load_rule_catalog()

    rules = {item["rule_id"]: item for item in catalog["patterns"]}
    assert set(rules) == set(BOOK_PATTERNS)
    assert len(rules) == len(catalog["patterns"])
    for rule in rules.values():
        assert rule["rule_set_version"] == catalog["rule_set_version"]
        assert rule["source_refs"]
        assert all(ref["source_status"] == "verified" for ref in rule["source_refs"])
        assert all(ref["printed_page"] > 0 and ref["pdf_page"] > 0 for ref in rule["source_refs"])
        assert all(ref["pdf_page"] == ref["printed_page"] + 15 for ref in rule["source_refs"])


def test_catalog_declares_confirmation_invalidation_and_target_stage():
    catalog = load_rule_catalog()

    for rule in catalog["patterns"]:
        assert rule["lifecycle"]["forming"]
        assert rule["lifecycle"]["confirmed_by"]
        assert rule["lifecycle"]["invalidated_by"]
        assert all(target["available_after"] in {"confirmed", "secondary_confirmed"}
                   for target in rule["targets"])


def test_duplicate_rule_id_reports_rule_and_field_path():
    catalog = load_rule_catalog()
    invalid = deepcopy(catalog)
    invalid["patterns"].append(deepcopy(invalid["patterns"][0]))

    with pytest.raises(RuleCatalogError) as error:
        validate_rule_catalog(invalid)

    assert any("patterns[12].rule_id" in item for item in error.value.errors)


def test_missing_lifecycle_field_reports_rule_and_field_path():
    catalog = load_rule_catalog()
    invalid = deepcopy(catalog)
    del invalid["patterns"][0]["lifecycle"]["confirmed_by"]

    with pytest.raises(RuleCatalogError) as error:
        validate_rule_catalog(invalid)

    assert any("w_bottom.lifecycle.confirmed_by" in item for item in error.value.errors)


def test_unknown_source_status_is_rejected():
    catalog = load_rule_catalog()
    invalid = deepcopy(catalog)
    invalid["patterns"][0]["source_refs"][0]["source_status"] = "unverified"

    with pytest.raises(RuleCatalogError) as error:
        validate_rule_catalog(invalid)

    assert any("w_bottom.source_refs[0].source_status" in item for item in error.value.errors)


def test_rule_json_is_valid_utf8_json():
    path = Path(__file__).parents[1] / "caisen" / "rules" / "patterns.v2.json"
    json.loads(path.read_text(encoding="utf-8"))


def test_active_catalog_entries_match_included_source_audit_rows():
    audit_path = Path(__file__).parents[1] / "docs" / "rule-source-audit.md"
    text = audit_path.read_text(encoding="utf-8")
    included_ids = set()
    for line in text.splitlines():
        match = re.match(r"\|\s*\d+\s+`([^`]+)`", line)
        if match and "**纳入" in line:
            included_ids.add(match.group(1))

    catalog_ids = {item["rule_id"] for item in load_rule_catalog()["patterns"]}
    assert catalog_ids == included_ids


def test_rule_cards_document_every_active_rule_id():
    cards_path = Path(__file__).parents[1] / "docs" / "rule-cards.md"
    cards = cards_path.read_text(encoding="utf-8")
    catalog_ids = {item["rule_id"] for item in load_rule_catalog()["patterns"]}

    assert all(f"`{rule_id}`" in cards for rule_id in catalog_ids)
    assert "rule-source-audit.md" in cards


def test_source_formula_examples_recompute_the_documented_targets():
    catalog = load_rule_catalog()
    formula_ids = set()
    covered_formula_ids = set()
    for rule in catalog["patterns"]:
        for target in rule["targets"]:
            formula_ids.add(target["formula_id"])
            if "source_example" not in target:
                continue
            covered_formula_ids.add(target["formula_id"])
            example = target["source_example"]
            values = example["inputs"]
            if target["formula_id"] == "neckline_height_projection":
                projected = values["break_price"] + values["neckline"] - values["structure_extreme"]
            elif target["formula_id"] == "head_neckline_projection":
                distance = abs(values["head"] - values["neckline"])
                projected = values["break_price"] + distance if rule["direction"] == "bullish" else values["break_price"] - distance
            elif target["formula_id"] == "first_wave_projection":
                distance = values["first_wave_high"] - values["first_wave_low"]
                projected = values["retracement_point"] + distance if rule["direction"] == "bullish" else values["retracement_point"] - distance
            elif target["formula_id"] == "triangle_width_projection":
                distance = values["triangle_high"] - values["triangle_low"]
                projected = values["break_price"] + distance if rule["direction"] == "bullish" else values["break_price"] - distance
            else:
                pytest.fail(f"No formula example evaluator for {target['formula_id']}")
            assert projected == example["expected_target"], rule["rule_id"]
    assert covered_formula_ids == formula_ids


def test_unsupported_target_formula_is_rejected_with_path():
    catalog = load_rule_catalog()
    invalid = deepcopy(catalog)
    invalid["patterns"][0]["targets"][0]["formula_id"] = "guess_from_chart"

    with pytest.raises(RuleCatalogError) as error:
        validate_rule_catalog(invalid)

    assert any("w_bottom.targets[0].formula_id" in item for item in error.value.errors)
