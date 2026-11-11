"""Different confirmed pressure lines must not consume each other's buy event."""
from copy import deepcopy

from caisen.main_bottoms import _same_neckline_boundary_phase


def _phase(slope=0.0, intercept=10.0):
    return {'line': {'slope': slope, 'intercept': intercept},
            'tests': [{'pivot_index': i, 'tolerance': .5} for i in (10, 30)],
            'consumed_index': 40, 'phase_tolerance': .5}


def test_new_supported_line_is_not_consumed_by_old_line_touch_tolerance():
    old = _phase()
    new = _phase(slope=-.001, intercept=10.2)
    new['tests'].append({'pivot_index': 60, 'tolerance': .5})
    assert not _same_neckline_boundary_phase(old, new, tick_size=.01)


def test_old_line_remains_consumed_when_only_touch_metadata_changes():
    old = _phase(slope=-.001, intercept=10.2)
    new = deepcopy(old)
    new['tests'].append({'pivot_index': 60, 'tolerance': .5})
    new['candidate_id'] = 'new-touch-group'
    assert _same_neckline_boundary_phase(old, new, tick_size=.01)


def test_material_slope_revision_is_not_hidden_by_nearby_anchor_levels():
    old = _phase(slope=-.001, intercept=10.2)
    new = _phase(slope=-.002, intercept=10.22)
    assert not _same_neckline_boundary_phase(old, new, tick_size=.01)
