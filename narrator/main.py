"""Entry point for the narrator app with Gemini narration."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv

from narrator.audio import AmbientPlayer, PCMPlayer, parse_pcm_format
from narrator.config import AppConfig, load_config
from narrator.context import (
    capture_context,
    capture_screen_png,
    capture_screen_png_peekaboo,
)
from narrator import gemini as gemini_module
from narrator.gemini import UsageSummary, generate_context_narration, generate_narration
from narrator.narration import SportsNarrator
from narrator.tts import stream_tts
from narrator.vision import VisionTracker, default_metrics, VisualMetrics


def _parse_duration(value: str) -> Optional[float]:
    """Parse a duration string like '2m', '90s', '1h', '5min' into seconds."""
    import re
    value = value.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|hour)?", value)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2) or "s"
    if unit in ("m", "min"):
        return num * 60
    if unit in ("h", "hr", "hour"):
        return num * 3600
    return num


def _override_config(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    from narrator.config import VALID_NARRATION_STYLES

    style = getattr(args, "style", None)
    if style and style.lower() in VALID_NARRATION_STYLES:
        style = style.lower()
    else:
        style = None

    if (
        style is None
        and args.interval is None
        and args.min_gap is None
        and not args.verbose
        and args.screen_index is None
        and args.display_id is None
    ):
        return config
    return replace(
        config,
        narration_style=style or config.narration_style,
        interval_sec=args.interval or config.interval_sec,
        min_gap_sec=args.min_gap or config.min_gap_sec,
        verbose=config.verbose or args.verbose,
        screen_index=(
            config.screen_index if args.screen_index is None else args.screen_index
        ),
        screencapture_display=(
            config.screencapture_display
            if args.display_id is None
            else args.display_id
        ),
    )


def _speak_line(line: str, config: AppConfig) -> None:
    pcm_format = parse_pcm_format(config.output_format)
    if not pcm_format:
        raise ValueError(
            "Only PCM output formats are supported for live playback. "
            "Set ELEVENLABS_OUTPUT_FORMAT to something like pcm_16000."
        )

    async def _run() -> None:
        with PCMPlayer(pcm_format) as player:
            await stream_tts(line, config, player.play_chunk)

    asyncio.run(_run())


def _log_verbose(config: AppConfig, message: str) -> None:
    if config.verbose:
        print(f"[Debug] {message}", file=sys.stderr)


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _line_fingerprint(line: str) -> str:
    return hashlib.sha1(line.encode("utf-8")).hexdigest()[:10]


def _token_set(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _too_similar_to_recent(line: str, recent: deque[str]) -> bool:
    if not recent:
        return False
    tokens = _token_set(line)
    if not tokens:
        return False
    for prev in recent:
        prev_tokens = _token_set(prev)
        if not prev_tokens:
            continue
        jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
        if jaccard >= 0.55:
            return True
    return False


@dataclass(frozen=True)
class _LaneProfile:
    name: str
    thinking_budget: int
    char_target: str
    min_chars: int
    min_words: int

    def as_overrides(self) -> Dict[str, Any]:
        return {
            "thinking_budget": self.thinking_budget,
            "char_target": self.char_target,
            "min_chars": self.min_chars,
            "min_words": self.min_words,
        }


_SHORT_LANE = _LaneProfile(
    name="short",
    thinking_budget=48,
    char_target="60-90",
    min_chars=50,
    min_words=8,
)

_LONG_LANE = _LaneProfile(
    name="long",
    thinking_budget=128,
    char_target="120-180",
    min_chars=80,
    min_words=14,
)


_STALE_PREFIXES = (
    "A moment ago in {app},",
    "Seconds back in {app},",
    "Quick rewind: {app} was on deck,",
    "Just before the switch, {app} had the spotlight,",
    "Flashback to {app},",
    "Earlier on {app},",
)


def _prefix_stale_context(line: str, app_name: str, turn_index: int) -> str:
    import re
    prefix = _STALE_PREFIXES[turn_index % len(_STALE_PREFIXES)]
    trimmed = line.lstrip(" ,:-")
    # Strip existing "In {App}," / "In the {App} thicket," leads to avoid double-context
    trimmed = re.sub(
        r"^(In|On|Over in|Back in|Still in)\s+[^,]{3,40},\s*",
        "",
        trimmed,
        count=1,
    )
    trimmed = trimmed[:1].upper() + trimmed[1:] if trimmed else trimmed
    return f"{prefix.format(app=app_name)} {trimmed}"


def _resolve_ambient_wav(config: AppConfig) -> Optional[str]:
    """Pick the ambient WAV: explicit env var > style_ambient > auto-detect."""
    if config.ambient_wav:
        return config.ambient_wav
    # Check style_ambient from YAML config
    style_path = config.style_ambient.get(config.narration_style)
    if style_path:
        return style_path
    # Auto-detect ambient/{style}.wav relative to package
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    auto = os.path.join(pkg_dir, "ambient", f"{config.narration_style}.wav")
    if os.path.isfile(auto):
        return auto
    return None


def _start_ambient(config: AppConfig) -> Optional[AmbientPlayer]:
    wav = _resolve_ambient_wav(config)
    if config.dry_run or not wav:
        return None
    pcm_format = parse_pcm_format(config.output_format)
    if not pcm_format:
        print(
            "Ambient audio requires a PCM output format (e.g., pcm_16000).",
            file=sys.stderr,
        )
        return None
    try:
        ambient = AmbientPlayer(
            wav,
            pcm_format,
            gain=config.ambient_gain,
            crossfade_sec=config.ambient_crossfade_sec,
        )
        ambient.__enter__()
        return ambient
    except Exception as exc:
        print(f"[Narrator] Ambient audio disabled: {exc}", file=sys.stderr)
        return None


@dataclass(frozen=True)
class _SpeechItem:
    line_id: int
    line: str
    line_hash: str
    capture_id: int
    capture_mono: float
    request_mono: float
    response_mono: float
    enqueue_mono: float
    app_name: str
    window_title: str
    lane: str = ""


class _SpeechQueue:
    def __init__(self, config: AppConfig, timeline: Optional["_Timeline"] = None) -> None:
        self._config = config
        self._queue: "queue.Queue[Optional[_SpeechItem]]" = queue.Queue(
            maxsize=config.queue_max
        )
        self._stop = threading.Event()
        self._recent_hashes: deque[str] = deque(maxlen=8)
        self._recent_lines: deque[str] = deque(maxlen=6)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._timeline = timeline

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            line_id = item.line_id
            line = item.line
            line_hash = item.line_hash
            start_mono = time.monotonic()
            age_ms = (start_mono - item.capture_mono) * 1000.0
            queue_ms = (start_mono - item.enqueue_mono) * 1000.0
            request_ms = (item.response_mono - item.request_mono) * 1000.0
            try:
                current_ctx = capture_context()
                current_app = (current_ctx.app_name or "").strip()
                item_app = (item.app_name or "").strip()
                if current_app and item_app and current_app != item_app:
                    if self._timeline:
                        self._timeline.log(
                            "tts_skip_stale",
                            line_id=line_id,
                            capture_id=item.capture_id,
                            hash=line_hash,
                            chars=len(line),
                            age_ms=f"{age_ms:.0f}",
                            app=_truncate(item_app),
                            current_app=_truncate(current_app),
                        )
                    continue
                if self._timeline:
                    extra = {}
                    if item.lane:
                        extra["lane"] = item.lane
                    self._timeline.log(
                        "tts_start",
                        line_id=line_id,
                        capture_id=item.capture_id,
                        hash=line_hash,
                        chars=len(line),
                        age_ms=f"{age_ms:.0f}",
                        queue_ms=f"{queue_ms:.0f}",
                        request_ms=f"{request_ms:.0f}",
                        app=_truncate(item_app),
                        window=_truncate(item.window_title),
                        current_app=_truncate(current_ctx.app_name),
                        current_window=_truncate(current_ctx.window_title),
                        **extra,
                    )
                _speak_line(line, self._config)
                end_mono = time.monotonic()
                duration_ms = (end_mono - start_mono) * 1000.0
                end_age_ms = (end_mono - item.capture_mono) * 1000.0
                if self._timeline:
                    self._timeline.log(
                        "tts_end",
                        line_id=line_id,
                        capture_id=item.capture_id,
                        hash=line_hash,
                        chars=len(line),
                        duration_ms=f"{duration_ms:.0f}",
                        age_ms=f"{end_age_ms:.0f}",
                    )
            except Exception as exc:
                print(f"[Narrator] TTS error: {exc}", file=sys.stderr)
                if self._timeline:
                    self._timeline.log(
                        "tts_error",
                        line_id=line_id,
                        capture_id=item.capture_id,
                        hash=line_hash,
                        error=str(exc),
                    )
            finally:
                self._queue.task_done()

    def update_config(self, config: AppConfig) -> None:
        """Update the config used for TTS (voice selection, etc.)."""
        self._config = config

    def enqueue(self, item: _SpeechItem) -> bool:
        if self._queue.full():
            return False
        if item.line_hash in self._recent_hashes:
            if self._timeline:
                self._timeline.log(
                    "queue_dedup",
                    line_id=item.line_id,
                    capture_id=item.capture_id,
                    hash=item.line_hash,
                    reason="exact",
                )
            return False
        if _too_similar_to_recent(item.line, self._recent_lines):
            if self._timeline:
                self._timeline.log(
                    "queue_dedup",
                    line_id=item.line_id,
                    capture_id=item.capture_id,
                    hash=item.line_hash,
                    reason="similar",
                )
            return False
        self._recent_hashes.append(item.line_hash)
        self._recent_lines.append(item.line)
        self._queue.put(item)
        if self._timeline:
            age_ms = (item.enqueue_mono - item.capture_mono) * 1000.0
            self._timeline.log(
                "queue_enqueue",
                line_id=item.line_id,
                capture_id=item.capture_id,
                hash=item.line_hash,
                size=self._queue.qsize(),
                chars=len(item.line),
                age_ms=f"{age_ms:.0f}",
                app=_truncate(item.app_name),
                window=_truncate(item.window_title),
            )
        return True

    def is_full(self) -> bool:
        return self._queue.full()

    def size(self) -> int:
        return self._queue.qsize()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)


class _CostTracker:
    def __init__(
        self, input_cost_per_m: float, output_cost_per_m: float, window_sec: float = 60.0
    ) -> None:
        self._input_cost_per_m = input_cost_per_m
        self._output_cost_per_m = output_cost_per_m
        self._window_sec = window_sec
        self._entries: deque[tuple[float, int, int]] = deque()

    def add(self, prompt_tokens: int, output_tokens: int) -> dict[str, float]:
        now = time.time()
        self._entries.append((now, prompt_tokens, output_tokens))
        self._trim(now)
        window_cost = self._compute_cost(
            sum(item[1] for item in self._entries),
            sum(item[2] for item in self._entries),
        )
        per_min = window_cost * (60.0 / self._window_sec)
        per_hour = per_min * 60.0
        per_call = self._compute_cost(prompt_tokens, output_tokens)
        return {
            "per_call": per_call,
            "per_min": per_min,
            "per_hour": per_hour,
            "window_cost": window_cost,
        }

    def _trim(self, now: float) -> None:
        while self._entries and now - self._entries[0][0] > self._window_sec:
            self._entries.popleft()

    def _compute_cost(self, prompt_tokens: int, output_tokens: int) -> float:
        return (
            (prompt_tokens * self._input_cost_per_m)
            + (output_tokens * self._output_cost_per_m)
        ) / 1_000_000.0


def _resolve_gemini_pricing(config: AppConfig) -> tuple[float, float] | None:
    if config.gemini_input_cost_per_m is not None and config.gemini_output_cost_per_m is not None:
        return config.gemini_input_cost_per_m, config.gemini_output_cost_per_m
    model = config.gemini_model.lower()
    if "gemini-3-flash-preview" in model or "gemini-3-flash" in model:
        return 0.50, 3.00
    if "gemini-2.5-flash-lite" in model:
        return 0.10, 0.40
    if "gemini-2.5-flash" in model:
        return 0.30, 2.50
    return None


class _Timeline:
    def __init__(self, enabled: bool, path: Optional[str] = None) -> None:
        self._enabled = enabled
        self._path = path
        self._start = time.monotonic()
        self._lock = threading.Lock()

    def log(self, event: str, **fields: object) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        elapsed = now - self._start
        payload = {
            "t": f"{elapsed:.3f}",
            "event": event,
            **fields,
        }
        line = " ".join(f"{key}={value}" for key, value in payload.items())
        if self._path:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        else:
            print(f"[Timeline] {line}", file=sys.stderr)


class _GeminiGate:
    def __init__(self, min_gap_sec: float, diff_threshold: float) -> None:
        self._min_gap_sec = min_gap_sec
        self._diff_threshold = diff_threshold
        self._last_app = ""
        self._last_window = ""
        self._last_line_time = 0.0
        self._idle_ticks = 0

    def evaluate(
        self,
        app_name: str,
        window_title: str,
        metrics: VisualMetrics,
        ignore_apps: list[str],
    ) -> tuple[bool, str]:
        now = time.time()
        if now - self._last_line_time < self._min_gap_sec:
            remaining = self._min_gap_sec - (now - self._last_line_time)
            return False, f"cooldown {remaining:.2f}s"

        if app_name in ignore_apps:
            return False, "ignored_app"

        app_changed = app_name != self._last_app
        window_changed = window_title and window_title != self._last_window
        active = metrics.activity >= self._diff_threshold

        if app_changed or window_changed or active:
            if app_changed:
                return True, "app_changed"
            if window_changed:
                return True, "window_changed"
            return True, "activity"

        self._idle_ticks += 1
        if self._idle_ticks >= 3:
            return True, "idle_tick"
        return False, "idle_wait"

    def mark_attempt(self) -> None:
        self._last_line_time = time.time()

    def mark_spoken(self, app_name: str, window_title: str) -> None:
        self._last_app = app_name
        self._last_window = window_title
        self._last_line_time = time.time()
        self._idle_ticks = 0

    def cooldown_remaining(self) -> float:
        now = time.time()
        remaining = self._min_gap_sec - (now - self._last_line_time)
        return max(0.0, remaining)

    def last_context(self) -> tuple[str, str]:
        return self._last_app, self._last_window


class _RuntimeState:
    """Mutable runtime state for settings that can change mid-session."""

    def __init__(self, config: AppConfig) -> None:
        self.style = config.narration_style
        self.profanity = config.profanity_level
        self.paused = False
        self.dual_lane = config.dual_lane
        self.turn_index = 0

    def apply_to_config(self, config: AppConfig) -> AppConfig:
        """Return a config with runtime-mutable fields overridden."""
        return replace(
            config,
            narration_style=self.style,
            profanity_level=self.profanity,
            dual_lane=self.dual_lane,
        )


def _check_control_file(
    control_path: str,
    state: _RuntimeState,
    timeline: "_Timeline",
    verbose: bool,
) -> list[str]:
    """Read and process commands from the control file. Returns list of events."""
    if not control_path:
        return []
    try:
        with open(control_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        os.remove(control_path)
    except FileNotFoundError:
        return []
    except OSError:
        return []
    if not raw:
        return []

    events: list[str] = []
    try:
        commands = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Support single command or list of commands
    if isinstance(commands, dict):
        commands = [commands]
    if not isinstance(commands, list):
        return []

    from narrator.config import VALID_NARRATION_STYLES

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        action = cmd.get("command", "").strip().lower()

        if action == "style":
            value = str(cmd.get("value", "")).strip().lower()
            if value in VALID_NARRATION_STYLES and value != state.style:
                old = state.style
                state.style = value
                events.append(f"style:{old}->{value}")
                timeline.log("control", command="style", old=old, new=value)
                if verbose:
                    print(f"[Control] Style changed: {old} -> {value}", file=sys.stderr)

        elif action == "profanity":
            value = str(cmd.get("value", "")).strip().lower()
            if value in {"off", "low", "high"} and value != state.profanity:
                old = state.profanity
                state.profanity = value
                events.append(f"profanity:{old}->{value}")
                timeline.log("control", command="profanity", old=old, new=value)
                if verbose:
                    print(f"[Control] Profanity changed: {old} -> {value}", file=sys.stderr)

        elif action == "pause":
            if not state.paused:
                state.paused = True
                events.append("paused")
                timeline.log("control", command="pause")
                if verbose:
                    print("[Control] Paused", file=sys.stderr)

        elif action == "resume":
            if state.paused:
                state.paused = False
                events.append("resumed")
                timeline.log("control", command="resume")
                if verbose:
                    print("[Control] Resumed", file=sys.stderr)

        elif action == "dual_lane":
            value = cmd.get("value")
            if isinstance(value, bool) and value != state.dual_lane:
                state.dual_lane = value
                events.append(f"dual_lane:{value}")
                timeline.log("control", command="dual_lane", value=value)
                if verbose:
                    print(f"[Control] Dual lane: {value}", file=sys.stderr)

    return events


def _write_status_file(
    status_path: str,
    state: _RuntimeState,
    config: AppConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write current runtime state to a status file for external readers."""
    if not status_path:
        return
    status = {
        "style": state.style,
        "profanity": state.profanity,
        "paused": state.paused,
        "dual_lane": state.dual_lane,
        "turn_index": state.turn_index,
        "pid": os.getpid(),
    }
    if extra:
        status.update(extra)
    try:
        tmp = status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f)
        os.replace(tmp, status_path)
    except OSError:
        pass


def _run_loop_gemini(
    config: AppConfig,
    control_file: Optional[str] = None,
    status_file: Optional[str] = None,
) -> None:
    """Run the narration loop using Gemini for generation."""
    runtime = _RuntimeState(config)
    gate = _GeminiGate(
        min_gap_sec=config.min_gap_sec, diff_threshold=config.diff_threshold
    )
    vision = VisionTracker()
    screen_warning_shown = False
    peekaboo_warning_shown = False
    gemini_error_shown = False
    gemini_fatal = False
    timeline = _Timeline(config.timeline, config.timeline_path)
    speech_queue = None if config.dry_run else _SpeechQueue(config, timeline=timeline)
    ambient = _start_ambient(config)
    pricing = _resolve_gemini_pricing(config)
    cost_tracker = (
        _CostTracker(pricing[0], pricing[1]) if pricing and config.verbose else None
    )
    last_fallback_time = 0.0
    fallback_min_gap = config.fallback_min_gap_sec
    history_lines: deque[str] = deque(maxlen=config.gemini_history_lines)
    context_history: deque[str] = deque(maxlen=config.gemini_history_contexts)
    turn_index = 0
    request_id = 0
    capture_id = 0
    dual_lane = config.dual_lane
    lane_cycle = itertools.cycle([_SHORT_LANE, _LONG_LANE]) if dual_lane else None
    _log_verbose(
        config,
        f"screen_index={config.screen_index} screencapture_display={config.screencapture_display}",
    )
    if dual_lane:
        _log_verbose(config, "dual_lane=enabled queue_max=2")
    timeline.log(
        "start",
        mode="gemini",
        interval=config.interval_sec,
        min_gap=config.min_gap_sec,
        dual_lane=dual_lane,
    )
    try:
        while True:
            # Check for live control commands
            control_events = _check_control_file(
                control_file or "", runtime, timeline, config.verbose,
            )
            if control_events:
                old_style = config.narration_style
                # Rebuild effective config with runtime overrides
                config = runtime.apply_to_config(config)
                # Update dual_lane and lane_cycle if changed
                if runtime.dual_lane != dual_lane:
                    dual_lane = runtime.dual_lane
                    lane_cycle = itertools.cycle([_SHORT_LANE, _LONG_LANE]) if dual_lane else None
                # Update speech queue config so TTS uses the new voice
                if speech_queue:
                    speech_queue.update_config(config)
                # Swap ambient track if style changed
                if config.narration_style != old_style:
                    new_wav = _resolve_ambient_wav(config)
                    if ambient:
                        ambient.__exit__(None, None, None)
                        ambient = None
                    if new_wav:
                        ambient = _start_ambient(config)
                _write_status_file(status_file or "", runtime, config)

            if runtime.paused:
                _write_status_file(status_file or "", runtime, config)
                time.sleep(0.5)
                continue

            if gemini_fatal:
                time.sleep(config.interval_sec)
                continue
            if speech_queue and speech_queue.is_full():
                _log_verbose(
                    config,
                    f"skip=queue_full size={speech_queue.size()}",
                )
                timeline.log("skip", reason="queue_full", size=speech_queue.size())
                time.sleep(0.1 if dual_lane else config.interval_sec)
                continue
            ctx = capture_context()
            app_name = (ctx.app_name or "Unknown App").strip()
            window_title = (ctx.window_title or "").strip()
            if not dual_lane:
                cooldown = gate.cooldown_remaining()
                last_app, last_window = gate.last_context()
                if (
                    cooldown > 0
                    and ctx.app_name == last_app
                    and ctx.window_title == last_window
                ):
                    _log_verbose(config, f"skip=cooldown {cooldown:.2f}s")
                    timeline.log("skip", reason="cooldown", remaining=f"{cooldown:.2f}")
                    time.sleep(config.interval_sec)
                    continue
            metrics = default_metrics()
            png_bytes = b""
            capture_start = time.monotonic()
            capture_source = ""
            capture_mono = 0.0

            try:
                try:
                    png_bytes = capture_screen_png_peekaboo(config.screen_index)
                    metrics = vision.analyze(png_bytes)
                    capture_source = "peekaboo"
                    _log_verbose(config, f"capture=peekaboo bytes={len(png_bytes)}")
                except Exception as exc:
                    if not peekaboo_warning_shown:
                        print(
                            f"Peekaboo capture failed, falling back to screencapture: {exc}",
                            file=sys.stderr,
                        )
                        peekaboo_warning_shown = True
                    png_bytes = capture_screen_png(config.screencapture_display)
                    metrics = vision.analyze(png_bytes)
                    capture_source = "screencapture"
                    _log_verbose(config, f"capture=screencapture bytes={len(png_bytes)}")
            except Exception as exc:
                vision.reset()
                if not screen_warning_shown:
                    print(f"Screen capture unavailable: {exc}", file=sys.stderr)
                    screen_warning_shown = True
                timeline.log("capture_error", error=str(exc))
                time.sleep(config.interval_sec)
                continue

            capture_mono = time.monotonic()
            capture_ms = (capture_mono - capture_start) * 1000.0
            capture_id += 1
            timeline.log(
                "capture",
                id=capture_id,
                source=capture_source,
                bytes=len(png_bytes),
                duration_ms=f"{capture_ms:.0f}",
                app=_truncate(app_name),
                window=_truncate(window_title),
                activity=f"{metrics.activity:.2f}",
                brightness=f"{metrics.brightness:.2f}",
            )
            context_snapshot = f"{app_name} | {window_title}".strip()
            if not context_history or context_history[-1] != context_snapshot:
                context_history.append(context_snapshot)
                timeline.log(
                    "context",
                    app=_truncate(app_name),
                    window=_truncate(window_title),
                )

            if dual_lane:
                # In dual mode, always request — pacing comes from queue fullness + API latency
                if app_name in config.ignore_apps:
                    _log_verbose(config, f"skip=ignored_app app={_truncate(app_name)}")
                    timeline.log("skip", reason="ignored_app", app=_truncate(app_name))
                    time.sleep(config.interval_sec)
                    continue
                should_request = True
                reason = "dual_lane"
            else:
                should_request, reason = gate.evaluate(
                    app_name, window_title, metrics, config.ignore_apps
                )
            if not should_request:
                _log_verbose(
                    config,
                    f"skip={reason} app={_truncate(app_name)} window={_truncate(window_title)} "
                    f"activity={metrics.activity:.2f} brightness={metrics.brightness:.2f}",
                )
                timeline.log(
                    "skip",
                    reason=reason,
                    app=_truncate(app_name),
                    window=_truncate(window_title),
                    activity=f"{metrics.activity:.2f}",
                )
                time.sleep(config.interval_sec)
                continue

            gate.mark_attempt()
            current_lane = next(lane_cycle) if lane_cycle else None
            lane_overrides = current_lane.as_overrides() if current_lane else None
            lane_name = current_lane.name if current_lane else ""
            request_id += 1
            request_mono = time.monotonic()
            capture_age_ms = (request_mono - capture_mono) * 1000.0
            _log_verbose(
                config,
                f"gemini_request model={config.gemini_model} bytes={len(png_bytes)} "
                f"app={_truncate(app_name)} window={_truncate(window_title)} "
                f"activity={metrics.activity:.2f} brightness={metrics.brightness:.2f}",
            )
            timeline.log(
                "gemini_request",
                id=request_id,
                capture_id=capture_id,
                model=config.gemini_model,
                bytes=len(png_bytes),
                capture_age_ms=f"{capture_age_ms:.0f}",
                app=_truncate(app_name),
                window=_truncate(window_title),
                **({"lane": lane_name} if lane_name else {}),
            )
            _log_verbose(
                config,
                f"history_lines={len(history_lines)} history_contexts={len(context_history)} "
                f"turn={turn_index}",
            )

            try:
                request_start = request_mono
                line, usage, source = generate_narration(
                    png_bytes,
                    app_name,
                    window_title,
                    config,
                    list(history_lines),
                    list(context_history),
                    turn_index,
                    verbose=config.verbose,
                    lane_overrides=lane_overrides,
                )
                response_mono = time.monotonic()
                request_ms = (response_mono - request_start) * 1000.0
                response_age_ms = (response_mono - capture_mono) * 1000.0
                timeline.log(
                    "gemini_response",
                    id=request_id,
                    capture_id=capture_id,
                    duration_ms=f"{request_ms:.0f}",
                    age_ms=f"{response_age_ms:.0f}",
                    attempts=usage.attempts,
                    source=source,
                    **({"lane": lane_name} if lane_name else {}),
                )
            except Exception as exc:
                message = str(exc)
                if not gemini_error_shown:
                    print(f"Gemini error: {message}", file=sys.stderr)
                    gemini_error_shown = True
                error_age_ms = (time.monotonic() - capture_mono) * 1000.0
                timeline.log(
                    "gemini_error",
                    id=request_id,
                    capture_id=capture_id,
                    error=message,
                    age_ms=f"{error_age_ms:.0f}",
                )
                if "404" in message and "models/" in message:
                    print(
                        "Gemini model not found. Set GEMINI_MODEL to a supported model "
                        "(for example gemini-3-flash-preview) or run ListModels.",
                        file=sys.stderr,
                    )
                    gemini_fatal = True
                now = time.time()
                if now - last_fallback_time >= fallback_min_gap:
                    fallback = generate_context_narration(
                        app_name,
                        window_title,
                        config,
                        list(history_lines),
                        list(context_history),
                        turn_index,
                    )
                    if fallback:
                        fallback_mono = time.monotonic()
                        age_ms = (fallback_mono - capture_mono) * 1000.0
                        line_hash = _line_fingerprint(fallback)
                        if config.dry_run:
                            print(fallback)
                            timeline.log(
                                "fallback",
                                id=request_id,
                                capture_id=capture_id,
                                chars=len(fallback),
                                age_ms=f"{age_ms:.0f}",
                                app=_truncate(app_name),
                            )
                            history_lines.append(fallback)
                            turn_index += 1
                            last_fallback_time = now
                            gate.mark_spoken(app_name, window_title)
                        elif speech_queue and speech_queue.enqueue(
                            _SpeechItem(
                                line_id=request_id,
                                line=fallback,
                                line_hash=line_hash,
                                capture_id=capture_id,
                                capture_mono=capture_mono,
                                request_mono=request_mono,
                                response_mono=fallback_mono,
                                enqueue_mono=fallback_mono,
                                app_name=app_name,
                                window_title=window_title,
                            )
                        ):
                            print(f"Announcer: {fallback}")
                            timeline.log(
                                "fallback",
                                id=request_id,
                                capture_id=capture_id,
                                chars=len(fallback),
                                age_ms=f"{age_ms:.0f}",
                                app=_truncate(app_name),
                            )
                            history_lines.append(fallback)
                            turn_index += 1
                            last_fallback_time = now
                            gate.mark_spoken(app_name, window_title)
                time.sleep(config.interval_sec)
                continue

            if cost_tracker and usage:
                if usage.prompt_tokens or usage.output_tokens:
                    costs = cost_tracker.add(usage.prompt_tokens, usage.output_tokens)
                    _log_verbose(
                        config,
                        "gemini_cost "
                        f"call=${costs['per_call']:.4f} "
                        f"window=${costs['window_cost']:.4f} "
                        f"rate=${costs['per_min']:.4f}/min "
                        f"${costs['per_hour']:.2f}/hr "
                        f"tokens_in={usage.prompt_tokens} "
                        f"tokens_out={usage.output_tokens}",
                    )
                    timeline.log(
                        "gemini_usage",
                        id=request_id,
                        capture_id=capture_id,
                        prompt=usage.prompt_tokens,
                        output=usage.output_tokens,
                        attempts=usage.attempts,
                        prompt_text=usage.prompt_text_tokens,
                        prompt_image=usage.prompt_image_tokens,
                    )

            if line:
                _log_verbose(config, f"gemini_line chars={len(line)}")
                line_hash = _line_fingerprint(line)
                timeline.log(
                    "gemini_line",
                    id=request_id,
                    capture_id=capture_id,
                    hash=line_hash,
                    chars=len(line),
                    source=source,
                    **({"lane": lane_name} if lane_name else {}),
                )
                enqueue_mono = time.monotonic()
                age_ms = (enqueue_mono - capture_mono) * 1000.0
                current_ctx = capture_context()
                current_app = (current_ctx.app_name or "").strip()
                current_window = (current_ctx.window_title or "").strip()
                context_changed = False
                if current_app and current_app != app_name:
                    context_changed = True
                elif (
                    current_app == app_name
                    and current_window
                    and window_title
                    and current_window != window_title
                ):
                    context_changed = True
                stale_age = age_ms / 1000.0
                age_exceeded = stale_age > config.max_stale_sec
                stale_reasons = []
                if age_exceeded:
                    stale_reasons.append("age")
                    if context_changed:
                        stale_reasons.append("context")
                if context_changed and not age_exceeded and config.narration_style != "nature":
                    line = _prefix_stale_context(line, app_name, turn_index)
                    timeline.log(
                        "stale_context_allow",
                        id=request_id,
                        capture_id=capture_id,
                        age_ms=f"{age_ms:.0f}",
                        app=_truncate(app_name),
                        window=_truncate(window_title),
                        current_app=_truncate(current_app),
                        current_window=_truncate(current_window),
                    )
                if config.drop_stale and stale_reasons:
                    timeline.log(
                        "stale_drop",
                        id=request_id,
                        capture_id=capture_id,
                        age_ms=f"{age_ms:.0f}",
                        reason=",".join(stale_reasons),
                        app=_truncate(app_name),
                        window=_truncate(window_title),
                        current_app=_truncate(current_app),
                        current_window=_truncate(current_window),
                    )
                    now = time.time()
                    if now - last_fallback_time >= fallback_min_gap:
                        if current_app or current_window:
                            current_snapshot = f"{current_app} | {current_window}".strip()
                            if (
                                current_snapshot
                                and (not context_history or context_history[-1] != current_snapshot)
                            ):
                                context_history.append(current_snapshot)
                            fallback = generate_context_narration(
                                current_app,
                                current_window,
                                config,
                                list(history_lines),
                                list(context_history),
                                turn_index,
                            )
                            if fallback:
                                fallback_hash = _line_fingerprint(fallback)
                                fallback_mono = time.monotonic()
                                fallback_age_ms = (fallback_mono - capture_mono) * 1000.0
                                if config.dry_run:
                                    print(fallback)
                                    timeline.log(
                                        "fallback",
                                        id=request_id,
                                        capture_id=capture_id,
                                        chars=len(fallback),
                                        age_ms=f"{fallback_age_ms:.0f}",
                                        app=_truncate(current_app),
                                    )
                                    history_lines.append(fallback)
                                    turn_index += 1
                                    last_fallback_time = now
                                    gate.mark_spoken(current_app, current_window)
                                elif speech_queue and speech_queue.enqueue(
                                    _SpeechItem(
                                        line_id=request_id,
                                        line=fallback,
                                        line_hash=fallback_hash,
                                        capture_id=capture_id,
                                        capture_mono=capture_mono,
                                        request_mono=request_mono,
                                        response_mono=fallback_mono,
                                        enqueue_mono=fallback_mono,
                                        app_name=current_app,
                                        window_title=current_window,
                                    )
                                ):
                                    print(f"Announcer: {fallback}")
                                    timeline.log(
                                        "fallback",
                                        id=request_id,
                                        capture_id=capture_id,
                                        chars=len(fallback),
                                        age_ms=f"{fallback_age_ms:.0f}",
                                        app=_truncate(current_app),
                                    )
                                    history_lines.append(fallback)
                                    turn_index += 1
                                    last_fallback_time = now
                                    gate.mark_spoken(current_app, current_window)
                    time.sleep(config.interval_sec)
                    continue
                if config.dry_run:
                    lane_label = f" [{lane_name}]" if lane_name else ""
                    print(f"{line}{lane_label}")
                    gate.mark_spoken(app_name, window_title)
                    history_lines.append(line)
                    turn_index += 1
                    if dual_lane:
                        continue  # skip sleep, immediately loop
                else:
                    if speech_queue and speech_queue.enqueue(
                        _SpeechItem(
                            line_id=request_id,
                            line=line,
                            line_hash=line_hash,
                            capture_id=capture_id,
                            capture_mono=capture_mono,
                            request_mono=request_mono,
                            response_mono=response_mono,
                            enqueue_mono=enqueue_mono,
                            app_name=app_name,
                            window_title=window_title,
                            lane=lane_name,
                        )
                    ):
                        print(f"Announcer: {line}")
                        timeline.log(
                            "enqueue",
                            id=request_id,
                            capture_id=capture_id,
                            hash=line_hash,
                            chars=len(line),
                            age_ms=f"{age_ms:.0f}",
                            **({"lane": lane_name} if lane_name else {}),
                        )
                        gate.mark_spoken(app_name, window_title)
                        history_lines.append(line)
                        turn_index += 1
                        if dual_lane:
                            continue  # skip sleep, immediately loop
                    else:
                        _log_verbose(config, "skip=queue_full")
                        timeline.log("skip", reason="queue_full")
            else:
                _log_verbose(config, "gemini_line empty")
                timeline.log("gemini_empty", id=request_id, capture_id=capture_id)
                now = time.time()
                if now - last_fallback_time >= fallback_min_gap:
                    fallback = generate_context_narration(
                        app_name,
                        window_title,
                        config,
                        list(history_lines),
                        list(context_history),
                        turn_index,
                    )
                    if fallback:
                        fallback_mono = time.monotonic()
                        age_ms = (fallback_mono - capture_mono) * 1000.0
                        line_hash = _line_fingerprint(fallback)
                        if config.dry_run:
                            print(fallback)
                            timeline.log(
                                "fallback",
                                id=request_id,
                                capture_id=capture_id,
                                chars=len(fallback),
                                age_ms=f"{age_ms:.0f}",
                                app=_truncate(app_name),
                            )
                            history_lines.append(fallback)
                            turn_index += 1
                            last_fallback_time = now
                            gate.mark_spoken(app_name, window_title)
                        elif speech_queue and speech_queue.enqueue(
                            _SpeechItem(
                                line_id=request_id,
                                line=fallback,
                                line_hash=line_hash,
                                capture_id=capture_id,
                                capture_mono=capture_mono,
                                request_mono=request_mono,
                                response_mono=fallback_mono,
                                enqueue_mono=fallback_mono,
                                app_name=app_name,
                                window_title=window_title,
                            )
                        ):
                            print(f"Announcer: {fallback}")
                            timeline.log(
                                "fallback",
                                id=request_id,
                                capture_id=capture_id,
                                chars=len(fallback),
                                age_ms=f"{age_ms:.0f}",
                                app=_truncate(app_name),
                            )
                            history_lines.append(fallback)
                            turn_index += 1
                            last_fallback_time = now
                            gate.mark_spoken(app_name, window_title)

            runtime.turn_index = turn_index
            _write_status_file(status_file or "", runtime, config)
            time.sleep(config.interval_sec)
    finally:
        if ambient:
            ambient.__exit__(None, None, None)
        if speech_queue:
            speech_queue.shutdown()
        # Clean up status file on exit
        if status_file:
            try:
                os.remove(status_file)
            except OSError:
                pass


def _run_loop_local(config: AppConfig) -> None:
    """Run the narration loop using local SportsNarrator."""
    narrator = SportsNarrator(
        min_gap_sec=config.min_gap_sec, diff_threshold=config.diff_threshold
    )
    vision = VisionTracker()
    screen_warning_shown = False
    ambient = _start_ambient(config)
    _log_verbose(
        config,
        f"screen_index={config.screen_index} screencapture_display={config.screencapture_display}",
    )

    try:
        while True:
            ctx = capture_context()
            metrics = default_metrics()

            try:
                png_bytes = capture_screen_png(config.screencapture_display)
                metrics = vision.analyze(png_bytes)
            except Exception as exc:
                vision.reset()
                if not screen_warning_shown:
                    print(f"Screen capture unavailable: {exc}", file=sys.stderr)
                    print(
                        "Narration will use app context only until Screen Recording "
                        "permission is granted.",
                        file=sys.stderr,
                    )
                    screen_warning_shown = True

            line = narrator.build_line(ctx, metrics, config.ignore_apps)
            if line:
                if config.dry_run:
                    print(line)
                else:
                    print(f"Announcer: {line}")
                    _speak_line(line, config)

            time.sleep(config.interval_sec)
    finally:
        if ambient:
            ambient.__exit__(None, None, None)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Live sports-style narrator for your computer activity."
    )
    parser.add_argument(
        "style",
        nargs="?",
        default=None,
        help="Narration style (or 'list' to show available styles)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available narration styles and exit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print lines only.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debug output.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local template-based narration (no AI calls).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between samples (overrides env).",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=None,
        help="Minimum seconds between spoken lines (overrides env).",
    )
    parser.add_argument(
        "--screen-index",
        type=int,
        default=None,
        help="Peekaboo screen index to capture (0 is usually the main display).",
    )
    parser.add_argument(
        "--display-id",
        type=int,
        default=None,
        help="screencapture display id to capture (macOS -D value).",
    )
    parser.add_argument(
        "--control-file",
        type=str,
        default=None,
        help="Path to a JSON control file for live settings changes.",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Path to write runtime status JSON for external readers.",
    )
    parser.add_argument(
        "-t", "--time",
        type=str,
        default=None,
        help="Auto-stop after duration, e.g. 2m, 5m, 90s, 1h.",
    )

    args = parser.parse_args(argv)

    # Style descriptions for display
    _STYLE_INFO = {
        "sports":     "Punchy play-by-play announcer, light roast",
        "nature":     "David Attenborough nature documentary",
        "horror":     "Creeping dread, ominous foreshadowing",
        "noir":       "Hard-boiled detective, rain-soaked cynicism",
        "reality_tv": "Reality TV confessional booth commentary",
        "asmr":       "Whispered meditation over mundane browsing",
        "wrestling":  "BAH GAWD maximum hype announcer",
    }

    def _print_styles() -> None:
        from narrator.config import _load_yaml_config, VALID_NARRATION_STYLES
        yaml_cfg = _load_yaml_config()
        voices = yaml_cfg.get("voices", {}) or {}
        ambient = yaml_cfg.get("ambient", {}) or {}
        print("\nAvailable narration styles:\n")
        for i, style in enumerate(sorted(VALID_NARRATION_STYLES), 1):
            desc = _STYLE_INFO.get(style, "")
            has_voice = "voice" if voices.get(style) else "     "
            has_ambient = "ambient" if ambient.get(style) else "       "
            print(f"  {i:2}. {style:<12} {desc:<50} [{has_voice}] [{has_ambient}]")
        print()

    def _pick_style() -> Optional[str]:
        from narrator.config import VALID_NARRATION_STYLES
        _print_styles()
        styles = sorted(VALID_NARRATION_STYLES)
        try:
            choice = input("Pick a style (number or name): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(styles):
                return styles[idx]
        if choice in VALID_NARRATION_STYLES:
            return choice
        print(f"Unknown style: {choice}", file=sys.stderr)
        return None

    # Handle --list or "list" as positional
    if args.list or (args.style and args.style.lower() == "list"):
        _print_styles()
        return 0

    # Interactive picker if no style given and stdin is a terminal
    if args.style is None and sys.stdin.isatty() and not args.dry_run:
        picked = _pick_style()
        if picked:
            args.style = picked
        # If they didn't pick, fall through to default from config

    # Parse and schedule auto-stop timer
    stop_timer = None
    if args.time:
        duration = _parse_duration(args.time)
        if duration is None or duration <= 0:
            print(f"Invalid duration: {args.time}", file=sys.stderr)
            return 2
        import signal
        def _timeout_handler(*_a: object) -> None:
            print(f"\n[Narrator] Time limit reached ({args.time}). Stopping.", file=sys.stderr)
            raise KeyboardInterrupt
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, duration)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path if dotenv_path else None)
    try:
        config = load_config(dry_run=args.dry_run, require_gemini=not args.local)
        config = _override_config(config, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if config.verbose:
            print(
                "[Debug] using_dotenv="
                + (dotenv_path if dotenv_path else "not_found"),
                file=sys.stderr,
            )
            print(f"[Debug] gemini_module={gemini_module.__file__}", file=sys.stderr)
            print(
                f"[Debug] gemini_template_version={getattr(gemini_module, 'TEMPLATE_VERSION', 'unknown')}",
                file=sys.stderr,
            )
            print(f"[Debug] config_model={config.gemini_model}", file=sys.stderr)
        if args.local:
            print("[Narrator] Using local template narration.")
            _run_loop_local(config)
        else:
            print("[Narrator] Using Gemini narration.")
            _run_loop_gemini(
                config,
                control_file=args.control_file,
                status_file=args.status_file,
            )
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
