"""PCM audio playback helper."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional
import threading
import wave

import numpy as np

import sounddevice as sd


_PCM_RE = re.compile(r"^pcm_(\d+)$")


@dataclass(frozen=True)
class PCMFormat:
    """PCM audio format settings."""

    sample_rate: int
    channels: int = 1
    sample_width: int = 2


def parse_pcm_format(output_format: str) -> Optional[PCMFormat]:
    """Parse a PCM output format like 'pcm_16000' into a PCMFormat."""

    match = _PCM_RE.match(output_format)
    if not match:
        return None
    return PCMFormat(sample_rate=int(match.group(1)))


class PCMPlayer:
    """Play raw PCM audio chunks through the default audio device."""

    def __init__(self, pcm_format: PCMFormat) -> None:
        self._pcm_format = pcm_format
        self._stream: Optional[sd.RawOutputStream] = None

    def __enter__(self) -> "PCMPlayer":
        self._stream = sd.RawOutputStream(
            samplerate=self._pcm_format.sample_rate,
            channels=self._pcm_format.channels,
            dtype="int16",
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def play_chunk(self, chunk: bytes) -> None:
        """Write PCM bytes to the audio output."""

        if not self._stream:
            raise RuntimeError("PCMPlayer is not started")
        if chunk:
            self._stream.write(chunk)


class AmbientPlayer:
    """Play a looping ambient WAV in the background."""

    def __init__(
        self,
        wav_path: str,
        pcm_format: PCMFormat,
        gain: float = 0.08,
        crossfade_sec: float = 0.25,
    ) -> None:
        self._wav_path = wav_path
        self._pcm_format = pcm_format
        self._gain = gain
        self._crossfade_sec = max(0.0, crossfade_sec)
        self._stream: Optional[sd.RawOutputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples: Optional[np.ndarray] = None
        self._index = 0
        self._fade_frames = 0

    def __enter__(self) -> "AmbientPlayer":
        with wave.open(self._wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            if channels != self._pcm_format.channels:
                raise ValueError("Ambient WAV must be mono.")
            if sample_width != self._pcm_format.sample_width:
                raise ValueError("Ambient WAV must be 16-bit PCM.")
            if sample_rate != self._pcm_format.sample_rate:
                raise ValueError(
                    "Ambient WAV sample rate must match output format."
                )
            frames = wf.readframes(wf.getnframes())

        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            raise ValueError("Ambient WAV is empty.")
        if self._gain != 1.0:
            samples = (samples.astype(np.float32) * self._gain).clip(
                -32768, 32767
            ).astype(np.int16)
        self._samples = samples
        self._fade_frames = int(self._pcm_format.sample_rate * self._crossfade_sec)

        self._stream = sd.RawOutputStream(
            samplerate=self._pcm_format.sample_rate,
            channels=self._pcm_format.channels,
            dtype="int16",
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _loop(self) -> None:
        if self._samples is None or self._stream is None:
            return
        chunk_size = int(self._pcm_format.sample_rate * 0.1)
        total = self._samples.size
        fade = self._fade_frames
        while not self._stop.is_set():
            end = self._index + chunk_size
            if end <= total:
                chunk = self._samples[self._index:end]
                self._index = end
            else:
                first = self._samples[self._index:]
                remaining = end - total
                second = self._samples[:remaining]
                if fade > 0 and first.size > 0 and second.size > 0:
                    fade_len = min(fade, first.size, second.size)
                    if fade_len > 0:
                        fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False)
                        fade_in = np.linspace(0.0, 1.0, fade_len, endpoint=False)
                        mixed = (
                            first[-fade_len:].astype(np.float32) * fade_out
                            + second[:fade_len].astype(np.float32) * fade_in
                        )
                        first = first.copy()
                        second = second.copy()
                        first[-fade_len:] = mixed.astype(np.int16)
                        second[:fade_len] = 0
                chunk = np.concatenate((first, second))
                self._index = remaining
            self._stream.write(chunk.tobytes())
