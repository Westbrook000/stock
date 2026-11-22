import importlib.util
from pathlib import Path
from types import SimpleNamespace

from caisen.schemas import Pattern


RUNNER = (Path(__file__).resolve().parents[1] / "research" / "experiments"
          / "structure_boundary_losses_20261112"
          / "validate_600108_positive_oct8_oct9_attempt06.py")
SPEC = importlib.util.spec_from_file_location("positive_attempt06", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pattern(pattern_id, instance_id):
    return Pattern(pattern_id=pattern_id, code="600108", kind="stock", timeframe="1d",
                   pattern_type="w_bottom", direction="bullish", state="confirmed",
                   formed_at="2024-02-06", confirmed_at=1,
                   evidence=[{"code": "joint_lifecycle", "value": {"instance_id": instance_id}}])


def test_session_snapshot_collector_accepts_none_live_pattern_and_real_patterns():
    live = _pattern("p-live", "instance-live")
    retired = _pattern("p-retired", "instance-retired")
    session = SimpleNamespace(active={"empty": {"pattern": None}, "live": {"pattern": live}},
                              retired_patterns={"retired": retired}, flag_formations={},
                              flag_instances={}, emitted_events=[],
                              _attach_session_lifecycle=lambda _pattern: None)

    collected = MODULE._session_pattern_snapshots(session)

    assert {(item.pattern_id, next(ev["value"]["instance_id"] for ev in item.evidence
                                   if ev["code"] == "joint_lifecycle")) for item in collected} == {
        ("p-live", "instance-live"), ("p-retired", "instance-retired")}


def test_session_snapshot_collector_converts_emitted_dict_payload():
    emitted = _pattern("p-emitted", "instance-emitted").to_dict()
    session = SimpleNamespace(active={}, retired_patterns={}, flag_formations={}, flag_instances={},
                              emitted_events=[{"pattern": emitted}],
                              _attach_session_lifecycle=lambda _pattern: None)

    collected = MODULE._session_pattern_snapshots(session)

    assert len(collected) == 1
    assert collected[0].pattern_id == "p-emitted"
    assert collected[0].evidence[0]["value"]["instance_id"] == "instance-emitted"
