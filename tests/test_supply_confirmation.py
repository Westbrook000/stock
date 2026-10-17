import unittest

import numpy as np

from wyckoff.analysis.supply_demand import Zone, zone_state_series


class SupplyConfirmationTests(unittest.TestCase):
    def test_zone_is_unavailable_until_departure_confirmed(self):
        high = np.full(20, 102.0)
        low = np.full(20, 99.5)
        close = np.full(20, 101.0)
        zone = Zone("D", 100.0, 99.0, 5, 100.0, 0.1, 2.0, 2, known_from=12)
        result = zone_state_series(high, low, close, np.ones(20), [zone])
        self.assertTrue(np.isnan(result["demand_top"][:12]).all())
        self.assertEqual(result["demand_top"][12], 100.0)
