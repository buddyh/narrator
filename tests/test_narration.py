import time
import unittest

from narrator.context import AppContext
from narrator.narration import SportsNarrator
from narrator.vision import VisualMetrics


class TestSportsNarrator(unittest.TestCase):
    def test_app_change_generates_line(self) -> None:
        narrator = SportsNarrator(min_gap_sec=0.0, diff_threshold=10.0)
        ctx = AppContext(app_name="Safari", window_title="", timestamp=time.time())
        metrics = VisualMetrics(brightness=0.5, activity=0.0, avg_color=(0, 0, 0))
        line = narrator.build_line(ctx, metrics, [])
        self.assertIsNotNone(line)
        self.assertTrue(line)

    def test_ignored_app_silences_line(self) -> None:
        narrator = SportsNarrator(min_gap_sec=0.0, diff_threshold=10.0)
        ctx = AppContext(app_name="Secrets", window_title="", timestamp=time.time())
        metrics = VisualMetrics(brightness=0.5, activity=0.0, avg_color=(0, 0, 0))
        line = narrator.build_line(ctx, metrics, ["Secrets"])
        self.assertIsNone(line)


if __name__ == "__main__":
    unittest.main()
