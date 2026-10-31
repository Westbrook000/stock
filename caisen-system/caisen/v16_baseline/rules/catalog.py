"""Load and validate the machine-readable 蔡森 rule catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).with_name("patterns.v2.json")
_DIRECTIONS = {"bullish", "bearish"}
_TARGET_STAGES = {"confirmed", "secondary_confirmed"}
_SOURCE_STATUS = {"verified"}
_FORMULA_IDS = {
    "neckline_height_projection",
    "head_neckline_projection",
    "first_wave_projection",
    "triangle_width_projection",
}


class RuleCatalogError(ValueError):
    """One or more catalog errors, each including an item and field path."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid rule catalog:\n" + "\n".join(self.errors))


def load_rule_catalog(path: Path | str | None = None) -> dict[str, Any]:
    """Read and validate a catalog JSON file without mutating its source."""
    source = Path(path) if path is not None else RULES_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleCatalogError([f"catalog: {exc}"]) from exc
    validate_rule_catalog(payload)
    return payload


def validate_rule_catalog(payload: Any) -> None:
    """Raise a path-specific error when the catalog contract is violated."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        raise RuleCatalogError(["catalog: expected an object"])
    version = payload.get("rule_set_version")
    if not isinstance(version, str) or not version.strip():
        errors.append("rule_set_version: expected a non-empty string")
    patterns = payload.get("patterns")
    if not isinstance(patterns, list):
        errors.append("patterns: expected a list")
        raise RuleCatalogError(errors)
    if not patterns:
        errors.append("patterns: expected at least one active rule")

    seen: set[str] = set()
    for index, rule in enumerate(patterns):
        base = f"patterns[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{base}: expected an object")
            continue
        rule_id = rule.get("rule_id")
        label = rule_id if isinstance(rule_id, str) and rule_id else base
        if not isinstance(rule_id, str) or not rule_id.strip():
            errors.append(f"{base}.rule_id: expected a non-empty string")
        elif rule_id in seen:
            errors.append(f"{base}.rule_id: duplicate rule ID {rule_id!r}")
        else:
            seen.add(rule_id)

        _required_text(rule, "name", label, errors)
        if rule.get("rule_set_version") != version:
            errors.append(f"{label}.rule_set_version: must match catalog version")
        if rule.get("audit_status") != "included":
            errors.append(f"{label}.audit_status: expected 'included'")
        if rule.get("direction") not in _DIRECTIONS:
            errors.append(f"{label}.direction: expected one of {sorted(_DIRECTIONS)}")

        for field in ("aliases", "engineering_parameters"):
            if not isinstance(rule.get(field), list):
                errors.append(f"{label}.{field}: expected a list")
        for field in ("topology",):
            _required_text(rule, field, label, errors)

        lifecycle = rule.get("lifecycle")
        if not isinstance(lifecycle, dict):
            errors.append(f"{label}.lifecycle: expected an object")
        else:
            forming = lifecycle.get("forming")
            if not isinstance(forming, list) or not forming or not all(
                isinstance(item, str) and item.strip() for item in forming
            ):
                errors.append(f"{label}.lifecycle.forming: expected non-empty text items")
            for field in ("confirmed_by", "invalidated_by"):
                _required_text(lifecycle, field, f"{label}.lifecycle", errors)

        volume = rule.get("volume")
        if not isinstance(volume, dict):
            errors.append(f"{label}.volume: expected an object")
        else:
            for field in ("required", "supporting"):
                if not isinstance(volume.get(field), list):
                    errors.append(f"{label}.volume.{field}: expected a list")

        targets = rule.get("targets")
        if not isinstance(targets, list):
            errors.append(f"{label}.targets: expected a list")
        else:
            for target_index, target in enumerate(targets):
                target_path = f"{label}.targets[{target_index}]"
                if not isinstance(target, dict):
                    errors.append(f"{target_path}: expected an object")
                    continue
                formula_id = target.get("formula_id")
                if formula_id not in _FORMULA_IDS:
                    errors.append(
                        f"{target_path}.formula_id: expected one of {sorted(_FORMULA_IDS)}"
                    )
                _required_text(target, "description", target_path, errors)
                if target.get("available_after") not in _TARGET_STAGES:
                    errors.append(
                        f"{target_path}.available_after: expected one of {sorted(_TARGET_STAGES)}"
                    )

        refs = rule.get("source_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{label}.source_refs: expected at least one verified reference")
        else:
            for ref_index, ref in enumerate(refs):
                ref_path = f"{label}.source_refs[{ref_index}]"
                if not isinstance(ref, dict):
                    errors.append(f"{ref_path}: expected an object")
                    continue
                for field in ("printed_page", "pdf_page"):
                    if not isinstance(ref.get(field), int) or ref[field] <= 0:
                        errors.append(f"{ref_path}.{field}: expected a positive page number")
                if ref.get("source_status") not in _SOURCE_STATUS:
                    errors.append(f"{ref_path}.source_status: expected 'verified'")
                if "figure" not in ref:
                    errors.append(f"{ref_path}.figure: include a figure caption or explicit null")

    if errors:
        raise RuleCatalogError(errors)


def _required_text(obj: dict[str, Any], field: str, prefix: str, errors: list[str]) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{field}: expected a non-empty string")
