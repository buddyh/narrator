"""Collect active app and screen information on macOS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Optional


@dataclass(frozen=True)
class AppContext:
    """Snapshot of the frontmost app and window."""

    app_name: str
    window_title: str
    timestamp: float


def _run_osascript(script: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_frontmost_app() -> str:
    """Return the name of the frontmost app, or an empty string on failure."""

    script = (
        "tell application \"System Events\" "
        "to get name of first application process whose frontmost is true"
    )
    value = _run_osascript(script)
    return value or ""


def get_frontmost_window_title() -> str:
    """Return the title of the front window, or an empty string on failure."""

    script = (
        "tell application \"System Events\" "
        "to tell (first application process whose frontmost is true) "
        "to get name of front window"
    )
    value = _run_osascript(script)
    return value or ""


_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def capture_screen_png(display_id: Optional[int] = None) -> bytes:
    """Capture the main display to PNG bytes using macOS screencapture."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "capture.png"

        command = ["/usr/sbin/screencapture", "-x", "-t", "png"]
        if display_id is not None:
            command.extend(["-D", str(display_id)])
        command.append(str(tmp_path))

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            hint = (
                "Check Screen Recording permission for your terminal or Python "
                "in System Settings > Privacy & Security."
            )
            detail = f" screencapture stderr: {stderr}" if stderr else ""
            raise RuntimeError(
                f"screencapture failed (code {result.returncode}). {hint}{detail}"
            )

        if not tmp_path.exists():
            raise RuntimeError("screencapture did not create an output file.")

        data = tmp_path.read_bytes()
        if not data.startswith(_PNG_HEADER):
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            hint = (
                "Check Screen Recording permission for your terminal or Python "
                "in System Settings > Privacy & Security."
            )
            detail = f" screencapture stderr: {stderr}" if stderr else ""
            raise RuntimeError(f"screencapture returned no PNG data. {hint}{detail}")

        return data


def capture_screen_png_peekaboo(screen_index: Optional[int] = None) -> bytes:
    """Capture the main display to PNG bytes using Peekaboo CLI."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "peekaboo_capture.png"
        try:
            command = [
                "peekaboo",
                "image",
                "--mode",
                "screen",
                "--format",
                "png",
                "--path",
                str(tmp_path),
            ]
            if screen_index is not None:
                command.extend(["--screen-index", str(screen_index)])

            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "peekaboo CLI not found on PATH. Install Peekaboo or adjust PATH."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            detail = f" peekaboo stderr: {stderr}" if stderr else ""
            raise RuntimeError(f"peekaboo capture failed (code {result.returncode}).{detail}")

        if not tmp_path.exists():
            raise RuntimeError("peekaboo capture did not create an output file.")

        data = tmp_path.read_bytes()
        if not data.startswith(_PNG_HEADER):
            raise RuntimeError("peekaboo returned non-PNG data.")

        return data


def capture_context() -> AppContext:
    """Capture the current active app and window title."""

    app_name = get_frontmost_app()
    window_title = get_frontmost_window_title()
    return AppContext(app_name=app_name, window_title=window_title, timestamp=time.time())
