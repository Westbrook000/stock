"""Horizontal day0 line regressions (replaces legacy band tests)."""
import unittest
import numpy as np
from wyckoff.analysis.support import SupportSeries
from wyckoff.analysis.zigzag import Pivot
from wyckoff.config import SupportConfig

class SupportLineTests(unittest.TestCase):
    def series(self, scale=1., length=30):
        pivots=[Pivot(2,100*scale,'low',5), Pivot(8,100.2*scale,'high',10),
                Pivot(13,110*scale,'high',17), Pivot(18,110.2*scale,'low',21)]
        return SupportSeries.compute(np.full(length,105.*scale),np.full(length,2.*scale),
                                     [p for p in pivots if p.confirm_idx<length],SupportConfig())
    def test_line_uses_confirmed_pivots_without_extra_count_filter(self):
        lines=self.series()
        self.assertTrue(np.isnan(lines.support[4]))
        self.assertEqual(lines.support[5],100.0)
        self.assertAlmostEqual(lines.support[10],100.1)
        self.assertAlmostEqual(lines.resistance[21],110.1)
    def test_prefix_is_unchanged_by_future_points(self):
        full=self.series(); prefix=self.series(length=19)
        np.testing.assert_allclose(full.support[:19],prefix.support,equal_nan=True)
        np.testing.assert_allclose(full.resistance[:19],prefix.resistance,equal_nan=True)
        self.assertTrue(all(p.known_from>=5 for p in full.levels))
        self.assertTrue(any(p.valid_to is not None for p in full.levels))
    def test_price_scale_does_not_change_clusters(self):
        normal=self.series(); scaled=self.series(100.)
        np.testing.assert_allclose(normal.support,scaled.support/100,equal_nan=True)
        np.testing.assert_allclose(normal.resistance,scaled.resistance/100,equal_nan=True)
        np.testing.assert_array_equal(normal.support_count,scaled.support_count)
