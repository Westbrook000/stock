"""Semantic checks for isolated attribution variants, without production edits."""
import ast
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import PROJECT, OUT, transform

def baseline_main():
    return (OUT / 'variants/no_prefix_cache/caisen/main_bottoms.py').read_text()


def boundary_matcher(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_same_neckline_boundary_phase')
    ns = {'_at': lambda line, index: line['slope'] * index + line['intercept']}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<phase matcher>', 'exec'), ns)
    return ns['_same_neckline_boundary_phase']


def test_distinct_supported_lines_are_not_consumed_together():
    main = baseline_main()
    daily = (PROJECT / 'caisen/daily_incremental.py').read_text()
    revised, _ = transform(main, daily, 'precise_boundary_identity')
    tests = [{'pivot_index': i, 'tolerance': .5} for i in (10, 30)]
    consumed = {'line': {'slope': 0., 'intercept': 10.}, 'tests': tests,
                'consumed_index': 40, 'phase_tolerance': .5}
    new = {'line': {'slope': -.001, 'intercept': 10.2}, 'tests': tests}
    assert boundary_matcher(main)(consumed, new)  # Reproduces broad old matching.
    assert not boundary_matcher(revised)(consumed, new)


def test_same_line_with_new_touch_metadata_still_matches():
    main = baseline_main()
    daily = (PROJECT / 'caisen/daily_incremental.py').read_text()
    revised, _ = transform(main, daily, 'precise_boundary_identity')
    a = {'line': {'slope': -.001, 'intercept': 10.2}, 'tests': []}
    b = {'line': dict(a['line']), 'tests': [{'pivot_index': 50, 'tolerance': .5}]}
    assert boundary_matcher(revised)(a, b)


def test_cache_control_does_not_change_boundary_or_geometry_source():
    main = baseline_main()
    daily = (PROJECT / 'caisen/daily_incremental.py').read_text()
    revised, d = transform(main, daily, 'no_prefix_cache')
    assert revised == main
    assert d.replace('    def _frame_for_current_prefix(self):\n'
                     '        self._prefix_frame_cache = None',
                     '    def _frame_for_current_prefix(self):', 1) == daily


def test_all_variants_are_syntactically_valid():
    from prepare import VARIANTS
    main = baseline_main()
    daily = (PROJECT / 'caisen/daily_incremental.py').read_text()
    for variant in VARIANTS:
        m, d = transform(main, daily, variant)
        compile(m, variant, 'exec')
        compile(d, variant, 'exec')
