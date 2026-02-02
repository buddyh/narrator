"""Simple visual analysis for screen change detection."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualMetrics:
    """Metrics derived from the current screen image."""

    brightness: float
    activity: float
    avg_color: Tuple[int, int, int]


class VisionTracker:
    """Track visual changes across frames."""

    def __init__(self) -> None:
        self._prev_small: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear the previous frame state."""

        self._prev_small = None

    def analyze(self, png_bytes: bytes) -> VisualMetrics:
        """Compute brightness, activity, and average color from PNG bytes."""

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        small = image.resize((64, 64), Image.BILINEAR)
        small_array = np.asarray(small, dtype=np.float32)
        gray = np.mean(small_array, axis=2)

        brightness = float(np.mean(gray) / 255.0)

        if self._prev_small is None:
            activity = 0.0
        else:
            diff = np.mean(np.abs(gray - self._prev_small))
            activity = float(diff)

        self._prev_small = gray

        avg_color = tuple(int(x) for x in np.mean(small_array, axis=(0, 1)))

        return VisualMetrics(brightness=brightness, activity=activity, avg_color=avg_color)


def default_metrics() -> VisualMetrics:
    """Return neutral metrics used when screen capture is unavailable."""

    return VisualMetrics(brightness=0.5, activity=0.0, avg_color=(127, 127, 127))
