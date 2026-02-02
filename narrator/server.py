#!/usr/bin/env python3
"""Simple socket server for narrator. Clawtto handles analysis directly."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

SOCKET_PATH = Path("/tmp/narrator_socket.sock")
PENDING_DIR = Path("/tmp/narrator_pending")
DONE_DIR = Path("/tmp/narrator_done")
STATE_FILE = Path("/tmp/narrator_state.json")
PENDING_DIR.mkdir(exist_ok=True)
DONE_DIR.mkdir(exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {"last_app": None, "last_window": None, "action_count": 0, "narrations": []}


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    addr = writer.get_extra_info('peername')
    
    try:
        # Read PNG size
        size_data = await reader.readexactly(4)
        png_size = int.from_bytes(size_data, 'big')
        if png_size <= 0:
            return
        
        # Read PNG and context
        png_bytes = await reader.readexactly(png_size)
        app_name = (await reader.readline()).decode("utf-8").strip()
        window_title = (await reader.readline()).decode("utf-8").strip()
        
        # Save for processing
        import time
        job_id = f"{int(time.time() * 1000)}"
        png_path = PENDING_DIR / f"{job_id}.png"
        png_path.write_bytes(png_bytes)
        
        # Load state for context
        state = load_state()
        
        # Generate dynamic prompt
        last_app = state.get("last_app", "unknown")
        action_count = state.get("action_count", 0)
        
        if app_name != last_app:
            activity = f"switched to {app_name}"
            state["action_count"] = action_count + 1
        else:
            activity = f"still in {app_name}"
        
        prompt = f'''You are a hilarious sports announcer narrating computer use. Be SNARKY, OBSERVATIONAL, COMEDIC.

CURRENT SCREEN: {app_name} - {window_title}
PREVIOUS: {last_app} ({activity})
SWITCHES: {state["action_count"]}

Write ONE short line (15-30 words) that:
- Describes what they're ACTUALLY DOING on screen
- Compares to what they were doing before
- Has personality - funny, running commentary
- References SPECIFIC content visible

EXAMPLES:
- "Here we go, BACK in Chrome... let's see what rabbit hole he falls into today!"
- "Oh! Switched to VS Code... wait, actually coding? Don't get used to it!"
- "There it is... doom-scrolling Twitter again. He really should work, but nah!"

ONE PUNCHY LINE, no markdown:'''

        # Call Clawtto properly - via shell with full context
        env = os.environ.copy()
        result = subprocess.run(
            ["bash", "-c", f'echo "{prompt}" | clawdbot agent --local'],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and len(l) > 10]
            narration = lines[0] if lines else f"And he's back in {app_name}!"
            narration = narration.replace("**", "").strip()
        else:
            narration = f"And he's using {app_name}!"
        
        # Update state
        state["last_app"] = app_name
        state["last_window"] = window_title
        state["narrations"].append(narration[-30:])
        state["narrations"] = state["narrations"][-5:]
        STATE_FILE.write_text(json.dumps(state, indent=2))
        
        # Cleanup
        png_path.unlink(missing_ok=True)
        
        # Send response
        writer.write(narration.encode("utf-8"))
        await writer.drain()
        print(f"[Server] {narration[:60]}...")
        
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    
    server = await asyncio.start_unix_server(handle_client, str(SOCKET_PATH))
    SOCKET_PATH.chmod(0o777)
    
    print(f"[Server] Listening on {SOCKET_PATH}")
    
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
