#!/usr/bin/env python3
"""File-based integration between narrator and Clawtto.

Workflow:
1. Narrator saves screenshot + context to /tmp/narrator_pending/
2. Clawtto monitors folder, analyzes images, writes narration to .txt
3. Narrator reads .txt, speaks it, cleans up
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

PENDING_DIR = Path("/tmp/narrator_pending")
OUTPUT_DIR = Path("/tmp/narrator_output")
PENDING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_capture(png_bytes: bytes, app_name: str, window_title: str) -> str:
    """Save screenshot and context, return the job ID."""
    import time
    job_id = f"{int(time.time() * 1000)}_{app_name.replace(' ', '_')}"
    
    (PENDING_DIR / f"{job_id}.png").write_bytes(png_bytes)
    (PENDING_DIR / f"{job_id}.ctx").write_text(f"{app_name}\n{window_title}")
    
    return job_id


def get_narration(job_id: str) -> str | None:
    """Check if Clawtto wrote narration for this job."""
    txt_path = OUTPUT_DIR / f"{job_id}.txt"
    if txt_path.exists():
        return txt_path.read_text().strip()
    return None


def cleanup(job_id: str) -> None:
    """Remove processed files."""
    for f in PENDING_DIR.glob(f"{job_id}.*"):
        f.unlink()
    for f in OUTPUT_DIR.glob(f"{job_id}.*"):
        f.unlink()


def call_clawtto(job_id: str) -> None:
    """Signal Clawtto to analyze the pending screenshot."""
    # Write a trigger file Clawtto can monitor
    trigger = PENDING_DIR / f"{job_id}.trigger"
    trigger.write_text(f"Analyze {job_id}")
    trigger.unlink()  # Clawtto will see the .png and .ctx files


if __name__ == "__main__":
    from narrator.context import capture_context, capture_screen_png

    ctx = capture_context()
    png = capture_screen_png()
    job_id = save_capture(png, ctx.app_name, ctx.window_title)
    
    # Signal Clawtto (this is a placeholder - actual integration depends on how Clawtto monitors)
    print(f"Saved capture {job_id}, awaiting Clawtto narration...")
    
    # In the full integration, Clawtto would write to OUTPUT_DIR/
    # For now, this shows the file structure
