"""ElevenLabs WebSocket streaming TTS."""

from __future__ import annotations

import base64
import json
import time
from typing import Callable, Optional, Any
import urllib.error
import urllib.request

import websockets

from narrator.config import AppConfig


AudioCallback = Callable[[bytes], None]


_VOICE_ID_CACHE: dict[tuple[str, str], str] = {}


def _resolve_voice_id(api_key: str, voice_name: str) -> str:
    cache_key = (api_key, voice_name.lower())
    cached = _VOICE_ID_CACHE.get(cache_key)
    if cached:
        return cached

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs voices error ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ElevenLabs voices connection error: {exc}") from exc

    data = json.loads(payload.decode("utf-8"))
    voices = data.get("voices") or []
    for voice in voices:
        name = str(voice.get("name", "")).strip()
        if name.lower() == voice_name.lower():
            voice_id = str(voice.get("voice_id", "")).strip()
            if voice_id:
                _VOICE_ID_CACHE[cache_key] = voice_id
                return voice_id

    available = ", ".join(
        sorted({str(voice.get("name", "")).strip() for voice in voices if voice.get("name")})
    )
    raise ValueError(
        f"ElevenLabs voice '{voice_name}' not found. Available voices: {available}"
        if available
        else f"ElevenLabs voice '{voice_name}' not found."
    )


async def stream_tts(
    text: str,
    config: AppConfig,
    on_audio_chunk: AudioCallback,
    on_event: Optional[Callable[[str], None]] = None,
) -> None:
    """Stream TTS audio for text and invoke a callback for each chunk."""

    if not text.strip():
        return
    if config.model_id.strip().startswith("eleven_v3"):
        raise ValueError(
            "Eleven v3 does not support WebSocket streaming. "
            "Use a v2.5 model (e.g., eleven_flash_v2_5) for streaming, "
            "or switch to the HTTP Create/Stream speech endpoints."
        )

    voice_id = config.voice_id
    # Per-style voice override
    style_vid = config.style_voice_ids.get(config.narration_style)
    if style_vid:
        voice_id = style_vid
    elif config.narration_style == "nature" and config.nature_voice_id:
        voice_id = config.nature_voice_id
    if not voice_id:
        if not config.voice_name:
            raise ValueError(
                "Missing ElevenLabs voice. Set ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE_NAME."
            )
        voice_id = _resolve_voice_id(config.api_key, config.voice_name)

    params = [f"model_id={config.model_id}", f"output_format={config.output_format}"]
    url = (
        "wss://api.elevenlabs.io/v1/text-to-speech/"
        f"{voice_id}/stream-input?" + "&".join(params)
    )

    voice_settings = {"stability": 0.4, "similarity_boost": 0.75}

    async with websockets.connect(url, ping_interval=20) as ws:
        init_payload = {
            "text": " ",
            "xi_api_key": config.api_key,
            "voice_settings": voice_settings,
        }
        _maybe_log_tts(config, "tts_send", init_payload, {"url": url})
        await ws.send(json.dumps(init_payload))

        gen_payload = {"text": text, "try_trigger_generation": True}
        _maybe_log_tts(config, "tts_send", gen_payload, {})
        await ws.send(json.dumps(gen_payload))
        end_payload = {"text": ""}
        _maybe_log_tts(config, "tts_send", end_payload, {})
        await ws.send(json.dumps(end_payload))

        async for message in ws:
            data = json.loads(message)
            _maybe_log_tts(config, "tts_recv", data, {})

            audio_payload = data.get("audio")
            if audio_payload:
                if not isinstance(audio_payload, str):
                    raise TypeError(
                        f"Unexpected audio payload type: {type(audio_payload)}"
                    )
                audio_bytes = base64.b64decode(audio_payload)
                if audio_bytes:
                    on_audio_chunk(audio_bytes)

            if "error" in data:
                _maybe_log_tts(config, "tts_error", data, {})
                raise RuntimeError(f"ElevenLabs error: {data['error']}")

            if data.get("isFinal") or data.get("final"):
                if on_event:
                    on_event("final")
                break


def _maybe_log_tts(
    config: AppConfig,
    event: str,
    payload: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    if not config.tts_trace_log:
        return
    ts = time.time()
    header = {"ts": f"{ts:.3f}", "event": event, **meta}
    safe_payload = dict(payload)
    if "xi_api_key" in safe_payload:
        safe_payload["xi_api_key"] = "***"
    if "audio" in safe_payload:
        audio_value = safe_payload.get("audio")
        if isinstance(audio_value, str):
            safe_payload["audio"] = f"<{len(audio_value)} chars redacted>"
    try:
        with open(config.tts_trace_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=True) + "\n")
            handle.write(json.dumps(safe_payload, ensure_ascii=True) + "\n")
    except OSError:
        return
