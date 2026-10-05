"""Checks for the star, friction, and phrase maps. No model call."""

import unittest

from pipeline.derived import (
    friction_band,
    is_retrieval_problem,
    memory_phrase,
    missing_detail,
    review_friction,
    segment_name,
    star_sentiment,
)


class DerivedMapTest(unittest.TestCase):
    def test_stars_cover_every_review(self):
        self.assertEqual(star_sentiment(1), "Frustrated")
        self.assertEqual(star_sentiment(2), "Frustrated")
        self.assertEqual(star_sentiment(3), "Neutral")
        self.assertEqual(star_sentiment(None), "Neutral")
        self.assertEqual(star_sentiment(4), "Positive")
        self.assertEqual(star_sentiment(5), "Positive")

    def test_friction_bands(self):
        self.assertEqual(friction_band(0), "0-20")
        self.assertEqual(friction_band(19), "0-20")
        self.assertEqual(friction_band(20), "20-40")
        self.assertEqual(friction_band(59), "40-60")
        self.assertEqual(friction_band(60), "60-80")
        self.assertEqual(friction_band(80), "80-100")
        self.assertEqual(friction_band(100), "80-100")

    def test_friction_stays_on_the_scale(self):
        low = review_friction({
            "rating": 5,
            "label": {"feeling": "relieved"},
            "extraction": {"outcome": "unknown"},
        })
        high = review_friction({
            "rating": 1,
            "label": {"feeling": "gave_up"},
            "extraction": {"outcome": "abandoned", "num_search_attempts": 3},
        })
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 100)

    def test_missing_detail_keeps_date_and_place_apart(self):
        self.assertEqual(missing_detail("exact dates of photos"), "The exact date")
        self.assertEqual(missing_detail("photo location"), "The place")
        self.assertEqual(missing_detail("specific photos"), "What the photo shows")

    def test_memory_phrase_drops_product_complaints(self):
        self.assertIsNone(memory_phrase("frustration with organization"))
        self.assertEqual(memory_phrase("friends"), "Friends")
        self.assertEqual(memory_phrase("family"), "Family")

    def test_segment_prefers_the_later_label(self):
        row = {
            "label": {"who": "parent", "pains": ["faces"]},
            "extraction": {"user_segment": "heavy_shooter"},
        }
        self.assertEqual(segment_name(row), "Parents")
        self.assertTrue(is_retrieval_problem(row))
        self.assertFalse(is_retrieval_problem({"label": {"pains": ["does_not_fit"]}}))


if __name__ == "__main__":
    unittest.main()
