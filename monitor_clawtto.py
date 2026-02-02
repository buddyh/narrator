#!/usr/bin/env python3
"""Monitor for new screenshots, track context, have Clawtto narrate dynamically."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

PENDING_DIR = Path("/tmp/narrator_pending")
DONE_DIR = Path("/tmp/narrator_done")
STATE_FILE = Path("/tmp/narrator_state.json")
PENDING_DIR.mkdir(exist_ok=True)
DONE_DIR.mkdir(exist_ok=True)


def load_state() -> dict:
    """Load previous state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {
        "last_app": None,
        "last_window": None,
        "last_activity": None,
        "action_count": 0,
        "narrations": []
    }


def save_state(state: dict) -> None:
    """Save state."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def monitor():
    """Poll for new screenshots and have Clawtto narrate dynamically."""
    print("[Narrator Monitor] Starting with context tracking...")
    
    state = load_state()
    state_dir = os.environ.get("CLAWDBOT_STATE_DIR", "")
    
    while True:
        # Check for new PNG files
        png_files = sorted(PENDING_DIR.glob("*.png"))
        
        for png_path in png_files:
            job_id = png_path.stem
            ctx_path = PENDING_DIR / f"{job_id}.ctx"
            done_path = DONE_DIR / f"{job_id}.txt"
            
            if done_path.exists():
                continue
            
            # Read context
            app_name = "Unknown"
            window_title = ""
            if ctx_path.exists():
                lines = ctx_path.read_text().strip().split("\n")
                app_name = lines[0] if lines else "Unknown"
                window_title = lines[1] if len(lines) > 1 else ""
            
            print(f"[Narrator Monitor] Analyzing {png_path.name}: {app_name}...")
            
            # Build dynamic prompt with context
            prompt = f"""You're a hilarious sports announcer narrating someone using their computer. This is COMEDY - be observational, snarky, and dynamic.

CURRENT SCREEN:
- App: {app_name}
- Window: {window_title}

PREVIOUS STATE (for comparison):
- Last app: {state.get('last_app', 'unknown')}
- Last action: {state.get('last_activity', 'unknown')}
- Total switches: {state.get('action_count', 0)}

YOUR JOB: Generate ONE short narration line (15-30 words) that:
1. OBSERVES what's actually ON the screen (specific content, not just app name)
2. COMPARES to previous state (did they switch? return? stay?)
3. HAS PERSONALITY - snarky, funny, observational
4. NARRATES their BEHAVIOR (doom-scrolling, working, distracted, etc.)
5. Is a running commentary, not just labels

EXAMPLES OF WHAT I WANT:
- "Here we go, he's BACK in Electron again... switching apps like it's his job!"
- "Oh! Making a move to Chrome... heading to x.com? Let's see what distraction awaits..."
- "There it is... doom-scrolling the feed. He really should get back to work..."
- "Wait, what's this? Looks like he's about to post something? Will anyone care?"
- "Consistency! Three hours in VS Code... actually, no wait, he's switching again!"
- "Chrome... YouTube... someone's avoiding their tasks today, huh?"

RULES:
- Be SNARKY but not mean
- Reference SPECIFIC content visible
- Notice PATTERNS (switching, returning, staying)
- Keep it SHORT but MEMORABLE
- NO markdown, just the line
- Use "he" or "they" - don't use "you"

Generate one punchy line:"""

            # Call Clawtto
            env = os.environ.copy()
            if state_dir:
                env["CLAWDBOT_STATE_DIR"] = state_dir
            
            try:
                result = subprocess.run(
                    ["clawdbot", "agent", "--local"],
                    input=f"Analyze {png_path} and generate a snarky sports-announcer narration. {prompt}",
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                )
                
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and len(l) > 10]
                    narration = lines[0] if lines else f"And they're back in {app_name}!"
                    narration = narration.replace("**", "").strip()
                else:
                    narration = f"And they're using {app_name}!"
            except Exception as e:
                print(f"[Narrator Monitor] Error: {e}")
                narration = f"And they're using {app_name}!"
            
            # Update state
            last_activity = state.get("last_activity", "")
            if app_name != state.get("last_app"):
                activity = f"switched to {app_name}"
                state["action_count"] = state.get("action_count", 0) + 1
            else:
                activity = f"still in {app_name}"
            
            state["last_app"] = app_name
            state["last_window"] = window_title
            state["last_activity"] = activity
            state["narrations"].append(narration[-50:])  # Keep last few
            state["narrations"] = state["narrations"][-5:]  # Keep last 5
            save_state(state)
            
            # Write result
            done_path.write_text(narration)
            print(f"[Narrator Monitor] Wrote: {narration[:60]}...")
            
            # Clean up
            try:
                png_path.unlink()
                ctx_path.unlink(missing_ok=True)
            except:
                pass
        
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(monitor())
