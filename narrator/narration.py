"""Generate sports-style narration lines."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import List, Optional

from narrator.context import AppContext
from narrator.vision import VisualMetrics


@dataclass
class NarrationState:
    """State used to avoid repetitive narration."""

    last_app: str = ""
    last_window: str = ""
    last_line_time: float = 0.0
    idle_ticks: int = 0


class SportsNarrator:
    """Create short, funny sports-announcer lines."""

    def __init__(self, min_gap_sec: float, diff_threshold: float) -> None:
        self._state = NarrationState()
        self._min_gap_sec = min_gap_sec
        self._diff_threshold = diff_threshold
        self._rng = random.Random()

    def build_line(
        self,
        ctx: AppContext,
        metrics: VisualMetrics,
        ignore_apps: List[str],
    ) -> Optional[str]:
        """Return a narration line or None if nothing should be spoken."""

        now = time.time()
        if now - self._state.last_line_time < self._min_gap_sec:
            return None

        app_name = ctx.app_name.strip() or "Unknown App"
        if app_name in ignore_apps:
            return None

        window_title = ctx.window_title.strip()
        title = window_title or "a mystery window"

        lighting = _lighting_tag(metrics.brightness)
        play_intensity = metrics.activity >= self._diff_threshold

        app_changed = app_name != self._state.last_app
        window_changed = title != self._state.last_window and window_title

        line: Optional[str] = None

        if app_changed:
            base_line = self._rng.choice(
                [
                    f"And they sprint onto the field with {app_name}!",
                    f"Big roster move: {app_name} just took center stage.",
                    f"Switching lanes to {app_name}, folks.",
                ]
            )
            line = _with_lighting(base_line, lighting)
        elif window_changed:
            base_line = self._rng.choice(
                [
                    f"New play on the board: {title} in {app_name}.",
                    f"They crack open {title} and the crowd leans in.",
                    f"Fresh tab alert: {title}.",
                ]
            )
            line = _with_lighting(base_line, lighting)
        elif play_intensity:
            base_line = self._rng.choice(
                [
                    f"Rapid fire clicks on {app_name}! The pace is wild.",
                    f"This screen is heating up in {app_name}.",
                    f"They are flying through it in {app_name}!",
                ]
            )
            line = _with_lighting(base_line, lighting)
        else:
            self._state.idle_ticks += 1
            if self._state.idle_ticks >= 3:
                base_line = self._rng.choice(
                    [
                        f"A strategic pause in {app_name}. The crowd waits.",
                        f"We have a calm moment in {app_name}.",
                        f"Quiet set here in {app_name}. A coach would call this poise.",
                    ]
                )
                line = _with_lighting(base_line, lighting)
                self._state.idle_ticks = 0

        if line:
            self._state.last_app = app_name
            self._state.last_window = title
            self._state.last_line_time = now
        return line


def _lighting_tag(brightness: float) -> str:
    if brightness < 0.3:
        return "Dark mode energy in the arena."
    if brightness > 0.7:
        return "Blindingly bright out there."
    return ""


def _with_lighting(line: str, lighting: str) -> str:
    if not lighting:
        return line
    return f"{line} {lighting}"
