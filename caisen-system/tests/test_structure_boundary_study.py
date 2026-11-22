import json
from pathlib import Path

import pytest

from scripts.run_structure_boundary_study import freeze_inputs, verify_frozen_snapshot


def _write_universe(path):
    path.write_text(json.dumps({
        "id": "fixed-v1", "date": "2026-11-01", "seed": 7,
        "counts": {"stock": 2, "etf": 1, "total": 3},
        "stocks": ["000001", "000002"], "etfs": ["510001"],
    }))


def test_freeze_inputs_keeps_fixed_members_and_source_revision(tmp_path):
    source = tmp_path / "source"
    for folder, code in (("stocks", "000001"), ("stocks", "000002"), ("etfs", "510001")):
        path = source / "data" / folder / (code + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,1\n")
    instrument_registry = source / "caisen-system" / "config" / "instruments.json"
    instrument_registry.parent.mkdir(parents=True)
    instrument_registry.write_text('{"instruments": []}')
    market_basis = source / "data" / "market_data_current" / ".market-data-basis.json"
    market_basis.parent.mkdir(parents=True)
    market_basis.write_text('{"adjustment_by_kind": {"stock": "qfq"}}')
    universe = tmp_path / "universe.json"
    _write_universe(universe)
    output = tmp_path / "snapshot"

    manifest = freeze_inputs(source, output, universe)

    assert manifest["counts"] == {"stock": 2, "etf": 1, "total": 3}
    assert verify_frozen_snapshot(output)["status"] == "valid"
    assert (output / "data" / "stocks" / "000001.csv").read_bytes() == (
        source / "data" / "stocks" / "000001.csv").read_bytes()
    assert (output / "caisen-system" / "config" / "instruments.json").is_file()
    assert (output / "data" / "market_data_current" / ".market-data-basis.json").is_file()


def test_freeze_inputs_refuses_to_overwrite_existing_snapshot(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "snapshot"
    output.mkdir()
    universe = tmp_path / "universe.json"
    _write_universe(universe)
    with pytest.raises(FileExistsError):
        freeze_inputs(source, output, universe)


def test_snapshot_verification_detects_market_file_revision_change(tmp_path):
    source = tmp_path / "source"
    path = source / "data" / "stocks" / "000001.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original")
    for folder, code in (("stocks", "000002"), ("etfs", "510001")):
        p = source / "data" / folder / (code + ".csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("data")
    registry = source / "caisen-system" / "config" / "instruments.json"
    registry.parent.mkdir(parents=True)
    registry.write_text('{"instruments": {}}')
    basis = source / "data" / "market_data_current" / ".market-data-basis.json"
    basis.parent.mkdir(parents=True)
    basis.write_text('{"adjustment_by_kind": {}}')
    universe = tmp_path / "universe.json"
    _write_universe(universe)
    snapshot = tmp_path / "snapshot"
    freeze_inputs(source, snapshot, universe)
    (snapshot / "data" / "stocks" / "000001.csv").write_text("changed")

    result = verify_frozen_snapshot(snapshot)

    assert result["status"] == "invalid"
    assert result["changed"][0]["code"] == "000001"
