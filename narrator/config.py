"""Configuration loading for the narrator app."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration values."""

    api_key: str
    voice_id: str
    voice_name: str
    model_id: str
    output_format: str
    gemini_api_key: str
    gemini_model: str
    gemini_temperature: float
    gemini_max_output_tokens: int
    gemini_timeout_sec: float
    gemini_min_chars: int
    gemini_min_words: int
    gemini_min_sentences: int
    gemini_max_retries: int
    gemini_history_lines: int
    gemini_history_contexts: int
    gemini_input_cost_per_m: Optional[float]
    gemini_output_cost_per_m: Optional[float]
    gemini_raw_log: Optional[str]
    gemini_trace_log: Optional[str]
    screen_index: Optional[int]
    screencapture_display: Optional[int]
    verbose: bool
    queue_max: int
    profanity_level: str
    ambient_wav: Optional[str]
    ambient_gain: float
    ambient_crossfade_sec: float
    debug_constraints: bool
    fallback_min_gap_sec: float
    max_stale_sec: float
    drop_stale: bool
    tts_trace_log: Optional[str]
    tts_log_audio: bool
    timeline: bool
    timeline_path: Optional[str]
    interval_sec: float
    min_gap_sec: float
    diff_threshold: float
    narration_style: str
    nature_voice_id: str
    style_voice_ids: Dict[str, str]
    style_ambient: Dict[str, str]
    ignore_apps: List[str]
    dry_run: bool
    dual_lane: bool


DEFAULT_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_OUTPUT_FORMAT = "pcm_16000"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_TEMPERATURE = 0.7
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 1024
DEFAULT_GEMINI_TIMEOUT_SEC = 15.0
DEFAULT_GEMINI_MIN_CHARS = 80
DEFAULT_GEMINI_MIN_WORDS = 14
DEFAULT_GEMINI_MIN_SENTENCES = 1
DEFAULT_GEMINI_MAX_RETRIES = 1
DEFAULT_GEMINI_HISTORY_LINES = 4
DEFAULT_GEMINI_HISTORY_CONTEXTS = 6
DEFAULT_QUEUE_MAX = 1
DEFAULT_PROFANITY_LEVEL = "high"
DEFAULT_NARRATION_STYLE = "sports"
VALID_NARRATION_STYLES = {
    "sports", "nature", "horror", "noir", "reality_tv",
    "asmr", "wrestling",
}
DEFAULT_AMBIENT_GAIN = 0.08
DEFAULT_AMBIENT_CROSSFADE_SEC = 1.5
DEFAULT_FALLBACK_MIN_GAP_SEC = 3.0
DEFAULT_MAX_STALE_SEC = 20.0
DEFAULT_DROP_STALE = True
DEFAULT_INTERVAL_SEC = 3.0
DEFAULT_MIN_GAP_SEC = 4.0
DEFAULT_DIFF_THRESHOLD = 8.0

CONFIG_PATH = Path.home() / ".narrator" / "config.yaml"


def _load_yaml_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML config file. Returns empty dict if not found."""
    p = path or CONFIG_PATH
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


def _split_list(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _read_positive_float(value: str, default: float, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return default
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0.")
    return parsed


def _read_positive_int(value: str, default: int, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0.")
    return parsed


def _read_nonnegative_int(value: str, default: int, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0.")
    return parsed


def _read_optional_int(value: str, name: str) -> Optional[int]:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0.")
    return parsed


def _read_optional_float(value: str, name: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0.")
    return parsed


def _read_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(dry_run: bool = False, require_gemini: bool = False) -> AppConfig:
    """Load configuration from ~/.narrator/config.yaml + environment variables.

    YAML provides defaults; env vars override everything.
    """
    yaml_cfg = _load_yaml_config()
    yaml_voices: Dict[str, str] = yaml_cfg.get("voices", {}) or {}
    yaml_ambient: Dict[str, str] = yaml_cfg.get("ambient", {}) or {}
    yaml_defaults: Dict[str, Any] = yaml_cfg.get("defaults", {}) or {}

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    voice_name = os.getenv("ELEVENLABS_VOICE_NAME", "").strip()
    if not voice_name:
        voice_name = os.getenv("ELEVENLABS_VOICE", "").strip()

    if not api_key and not dry_run:
        raise ValueError(
            "Missing ELEVENLABS_API_KEY. Set it in your environment or .env file."
        )
    if not voice_id and not voice_name and not dry_run:
        raise ValueError(
            "Missing ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE_NAME. "
            "Set one in your environment or .env file."
        )

    model_id = os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID).strip()
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT).strip()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        gemini_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if require_gemini and not gemini_api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY. Set it in your environment or .env file."
        )

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    gemini_temperature = float(
        os.getenv("GEMINI_TEMPERATURE", str(DEFAULT_GEMINI_TEMPERATURE))
    )
    gemini_max_output_tokens = _read_positive_int(
        os.getenv("GEMINI_MAX_OUTPUT_TOKENS", str(DEFAULT_GEMINI_MAX_OUTPUT_TOKENS)),
        DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
        "GEMINI_MAX_OUTPUT_TOKENS",
    )
    gemini_timeout_sec = _read_positive_float(
        os.getenv("GEMINI_TIMEOUT_SEC", str(DEFAULT_GEMINI_TIMEOUT_SEC)),
        DEFAULT_GEMINI_TIMEOUT_SEC,
        "GEMINI_TIMEOUT_SEC",
    )
    gemini_min_chars = _read_nonnegative_int(
        os.getenv("GEMINI_MIN_CHARS", str(DEFAULT_GEMINI_MIN_CHARS)),
        DEFAULT_GEMINI_MIN_CHARS,
        "GEMINI_MIN_CHARS",
    )
    gemini_min_words = _read_nonnegative_int(
        os.getenv("GEMINI_MIN_WORDS", str(DEFAULT_GEMINI_MIN_WORDS)),
        DEFAULT_GEMINI_MIN_WORDS,
        "GEMINI_MIN_WORDS",
    )
    gemini_min_sentences = _read_positive_int(
        os.getenv("GEMINI_MIN_SENTENCES", str(DEFAULT_GEMINI_MIN_SENTENCES)),
        DEFAULT_GEMINI_MIN_SENTENCES,
        "GEMINI_MIN_SENTENCES",
    )
    gemini_max_retries = _read_nonnegative_int(
        os.getenv("GEMINI_MAX_RETRIES", str(DEFAULT_GEMINI_MAX_RETRIES)),
        DEFAULT_GEMINI_MAX_RETRIES,
        "GEMINI_MAX_RETRIES",
    )
    gemini_history_lines = _read_nonnegative_int(
        os.getenv("GEMINI_HISTORY_LINES", str(DEFAULT_GEMINI_HISTORY_LINES)),
        DEFAULT_GEMINI_HISTORY_LINES,
        "GEMINI_HISTORY_LINES",
    )
    gemini_history_contexts = _read_nonnegative_int(
        os.getenv("GEMINI_HISTORY_CONTEXTS", str(DEFAULT_GEMINI_HISTORY_CONTEXTS)),
        DEFAULT_GEMINI_HISTORY_CONTEXTS,
        "GEMINI_HISTORY_CONTEXTS",
    )
    gemini_input_cost_per_m = _read_optional_float(
        os.getenv("GEMINI_INPUT_COST_PER_M", ""),
        "GEMINI_INPUT_COST_PER_M",
    )
    gemini_output_cost_per_m = _read_optional_float(
        os.getenv("GEMINI_OUTPUT_COST_PER_M", ""),
        "GEMINI_OUTPUT_COST_PER_M",
    )
    gemini_raw_log = os.getenv("NARRATOR_GEMINI_RAW_LOG", "").strip() or None
    gemini_trace_log = os.getenv("NARRATOR_GEMINI_TRACE_LOG", "").strip() or None
    screen_index = _read_optional_int(
        os.getenv("NARRATOR_SCREEN_INDEX", ""),
        "NARRATOR_SCREEN_INDEX",
    )
    screencapture_display = _read_optional_int(
        os.getenv("NARRATOR_SCREENCAPTURE_DISPLAY", ""),
        "NARRATOR_SCREENCAPTURE_DISPLAY",
    )

    verbose = _read_bool(os.getenv("NARRATOR_VERBOSE", ""))
    queue_max = _read_positive_int(
        os.getenv("NARRATOR_QUEUE_MAX", str(DEFAULT_QUEUE_MAX)),
        DEFAULT_QUEUE_MAX,
        "NARRATOR_QUEUE_MAX",
    )
    yaml_profanity = str(yaml_defaults.get("profanity", "")).strip().lower()
    profanity_level = os.getenv(
        "NARRATOR_PROFANITY", yaml_profanity or str(DEFAULT_PROFANITY_LEVEL)
    ).strip().lower()
    if profanity_level not in {"off", "low", "high"}:
        profanity_level = DEFAULT_PROFANITY_LEVEL
    ambient_wav = os.getenv("NARRATOR_AMBIENT_WAV", "").strip()
    if not ambient_wav:
        ambient_wav = None
    ambient_gain = _read_positive_float(
        os.getenv("NARRATOR_AMBIENT_GAIN", str(DEFAULT_AMBIENT_GAIN)),
        DEFAULT_AMBIENT_GAIN,
        "NARRATOR_AMBIENT_GAIN",
    )
    ambient_crossfade_sec = _read_positive_float(
        os.getenv(
            "NARRATOR_AMBIENT_CROSSFADE_SEC", str(DEFAULT_AMBIENT_CROSSFADE_SEC)
        ),
        DEFAULT_AMBIENT_CROSSFADE_SEC,
        "NARRATOR_AMBIENT_CROSSFADE_SEC",
    )
    debug_constraints = _read_bool(os.getenv("NARRATOR_DEBUG_CONSTRAINTS", ""))
    fallback_min_gap_sec = _read_positive_float(
        os.getenv("NARRATOR_FALLBACK_MIN_GAP", str(DEFAULT_FALLBACK_MIN_GAP_SEC)),
        DEFAULT_FALLBACK_MIN_GAP_SEC,
        "NARRATOR_FALLBACK_MIN_GAP",
    )
    max_stale_sec = _read_positive_float(
        os.getenv("NARRATOR_MAX_STALE_SEC", str(DEFAULT_MAX_STALE_SEC)),
        DEFAULT_MAX_STALE_SEC,
        "NARRATOR_MAX_STALE_SEC",
    )
    drop_stale_env = os.getenv("NARRATOR_DROP_STALE", "").strip()
    drop_stale = DEFAULT_DROP_STALE if not drop_stale_env else _read_bool(drop_stale_env)
    tts_trace_log = os.getenv("NARRATOR_TTS_TRACE_LOG", "").strip() or None
    tts_log_audio = _read_bool(os.getenv("NARRATOR_TTS_LOG_AUDIO", ""))
    timeline = _read_bool(os.getenv("NARRATOR_TIMELINE", ""))
    timeline_path = os.getenv("NARRATOR_TIMELINE_PATH", "").strip() or None

    interval_sec = _read_positive_float(
        os.getenv("NARRATOR_INTERVAL_SEC", str(DEFAULT_INTERVAL_SEC)),
        DEFAULT_INTERVAL_SEC,
        "NARRATOR_INTERVAL_SEC",
    )
    min_gap_sec = _read_positive_float(
        os.getenv("NARRATOR_MIN_GAP_SEC", str(DEFAULT_MIN_GAP_SEC)),
        DEFAULT_MIN_GAP_SEC,
        "NARRATOR_MIN_GAP_SEC",
    )
    diff_threshold = float(
        os.getenv("NARRATOR_DIFF_THRESHOLD", str(DEFAULT_DIFF_THRESHOLD))
    )

    yaml_style = str(yaml_defaults.get("style", "")).strip().lower()
    narration_style = os.getenv(
        "NARRATOR_STYLE", yaml_style or DEFAULT_NARRATION_STYLE
    ).strip().lower()
    if narration_style not in VALID_NARRATION_STYLES:
        narration_style = DEFAULT_NARRATION_STYLE

    nature_voice_id = os.getenv("NARRATOR_NATURE_VOICE_ID", "").strip()

    # Build style voice IDs: YAML first, then env vars override
    style_voice_ids: Dict[str, str] = {}
    for s, vid in yaml_voices.items():
        s_lower = str(s).strip().lower()
        if s_lower in VALID_NARRATION_STYLES and vid:
            style_voice_ids[s_lower] = str(vid).strip()
    for s in VALID_NARRATION_STYLES:
        env_key = f"NARRATOR_{s.upper()}_VOICE_ID"
        vid = os.getenv(env_key, "").strip()
        if vid:
            style_voice_ids[s] = vid
    # Backward compat: nature_voice_id -> style_voice_ids["nature"]
    if nature_voice_id and "nature" not in style_voice_ids:
        style_voice_ids["nature"] = nature_voice_id

    # Build style ambient paths from YAML
    style_ambient: Dict[str, str] = {}
    for s, path in yaml_ambient.items():
        s_lower = str(s).strip().lower()
        if s_lower in VALID_NARRATION_STYLES and path:
            expanded = os.path.expanduser(str(path).strip())
            if os.path.isfile(expanded):
                style_ambient[s_lower] = expanded

    ignore_apps = _split_list(os.getenv("NARRATOR_IGNORE_APPS", ""))

    dual_lane = _read_bool(os.getenv("NARRATOR_DUAL_LANE", ""))
    if dual_lane:
        queue_max = max(queue_max, 2)

    return AppConfig(
        api_key=api_key,
        voice_id=voice_id,
        voice_name=voice_name,
        model_id=model_id,
        output_format=output_format,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        gemini_temperature=gemini_temperature,
        gemini_max_output_tokens=gemini_max_output_tokens,
        gemini_timeout_sec=gemini_timeout_sec,
        gemini_min_chars=gemini_min_chars,
        gemini_min_words=gemini_min_words,
        gemini_min_sentences=gemini_min_sentences,
        gemini_max_retries=gemini_max_retries,
        gemini_history_lines=gemini_history_lines,
        gemini_history_contexts=gemini_history_contexts,
        gemini_input_cost_per_m=gemini_input_cost_per_m,
        gemini_output_cost_per_m=gemini_output_cost_per_m,
        gemini_raw_log=gemini_raw_log,
        gemini_trace_log=gemini_trace_log,
        screen_index=screen_index,
        screencapture_display=screencapture_display,
        verbose=verbose,
        queue_max=queue_max,
        profanity_level=profanity_level,
        ambient_wav=ambient_wav,
        ambient_gain=ambient_gain,
        ambient_crossfade_sec=ambient_crossfade_sec,
        debug_constraints=debug_constraints,
        fallback_min_gap_sec=fallback_min_gap_sec,
        max_stale_sec=max_stale_sec,
        drop_stale=drop_stale,
        tts_trace_log=tts_trace_log,
        tts_log_audio=tts_log_audio,
        timeline=timeline,
        timeline_path=timeline_path,
        interval_sec=interval_sec,
        min_gap_sec=min_gap_sec,
        diff_threshold=diff_threshold,
        narration_style=narration_style,
        nature_voice_id=nature_voice_id,
        style_voice_ids=style_voice_ids,
        style_ambient=style_ambient,
        ignore_apps=ignore_apps,
        dry_run=dry_run,
        dual_lane=dual_lane,
    )
