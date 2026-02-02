#!/usr/bin/env python3
"""Call Clawtto to generate narration from a screenshot."""

import subprocess
import sys
import tempfile
from pathlib import Path


def generate_narration(png_bytes: bytes, app_name: str, window_title: str) -> str:
    """Send screenshot to Clawtto and get back narration."""

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name

    try:
        # Call Clawtto via image analysis
        result = subprocess.run(
            ["clawdbot", "image", str(tmp_path), 
             f"Analyze this screenshot and write a short, funny sports-announcer style narration line. App: {app_name}, Window: {window_title}. Keep it short (1 sentence, punchy), sports energy, reference what's shown. Just return the line, no markdown."],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    from narrator.context import capture_context, capture_screen_png

    ctx = capture_context()
    png = capture_screen_png()
    line = generate_narration(png, ctx.app_name, ctx.window_title)
    if line:
        print(line)
    else:
        print("Failed to get narration from Clawtto", file=sys.stderr)
        sys.exit(1)
