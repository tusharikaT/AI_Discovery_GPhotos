"""Unit checks for the locked opportunity formula."""

import unittest

from pipeline.metrics import frequency_score, impact_score, opportunity_band


class MetricsFormulaTest(unittest.TestCase):
    def test_frequency_is_share_times_ten(self):
        self.assertEqual(frequency_score(25, 100), 2.5)
        self.assertEqual(frequency_score(0, 0), 0.0)

    def test_impact_uses_locked_weights(self):
        # 0.4*10 + 0.3*(1*10) + 0.3*9 = 4 + 3 + 2.7 = 9.7
        self.assertEqual(impact_score(10, 1.0, 9), 9.7)

    def test_opportunity_bands(self):
        self.assertEqual(opportunity_band(4.5, 6.7), "Core Strategic Bet")
        self.assertEqual(opportunity_band(3.6, 5.9), "Critical Safety Net")
        self.assertEqual(opportunity_band(4.0, 4.0), "Everyday Papercut")
        self.assertEqual(opportunity_band(1.4, 6.0), "Low-Priority Nuance")


if __name__ == "__main__":
    unittest.main()
