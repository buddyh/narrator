---
name: screen-narrator
description: "Live narration of your screen activity. Starts a background narrator that captures your screen, generates commentary via Gemini, and speaks it aloud with ElevenLabs TTS. Supports 7 narration styles (sports, nature, horror, noir, reality_tv, asmr, wrestling) with live switching."
homepage: https://github.com/buddyh/narrator
metadata: {"clawdbot":{"emoji":"","os":["darwin"],"requires":{"bins":["python3","tmux"],"envs":["GEMINI_API_KEY","ELEVENLABS_API_KEY"]}}}
---

# Screen Narrator

Live narration of your screen activity. Uses Gemini for vision + commentary and ElevenLabs for TTS. 7 styles from sports play-by-play to reality TV confessionals.

## Setup

Install dependencies:
```bash
cd ~/narrator && pip install -r requirements.txt
```

Required environment variables (in `~/narrator/.env` or exported):
- `GEMINI_API_KEY` - Google Gemini API key
- `ELEVENLABS_API_KEY` - ElevenLabs API key

Per-style voice IDs and ambient tracks are configured in `~/.narrator/config.yaml`.

## Screen Recording Permissions

**Clawdbot macOS app**: Screen recording works automatically — the app holds the TCC grant via `CGRequestScreenCaptureAccess()`. If it stops working after a rebuild, reset and relaunch:
```bash
sudo tccutil reset ScreenCapture com.clawdbot.mac
```

**Headless / node host**: Screen recording requires the macOS app path. Headless nodes can't hold TCC permissions. Run the narrator in a tmux session instead (see below).

## Starting the Narrator

### Via Clawdbot macOS app (has screen recording permission)

```bash
python -m narrator                    # interactive style picker
python -m narrator horror             # specific style
python -m narrator wrestling -t 5m    # auto-stop after 5 minutes
python -m narrator --list             # show available styles
```

### Headless / via Clawdbot agent (no screen recording permission)

Launch the narrator in a tmux session with control files. All mode changes happen via the control file — no send-keys needed.

**Start:**
```bash
tmux new-session -d -s narrator "cd ~/narrator && python -m narrator sports --control-file /tmp/narrator-ctl.json --status-file /tmp/narrator-status.json"
```

**Change style:**
```bash
echo '{"command": "style", "value": "horror"}' > /tmp/narrator-ctl.json
```

**Change profanity:**
```bash
echo '{"command": "profanity", "value": "low"}' > /tmp/narrator-ctl.json
```

**Pause / Resume:**
```bash
echo '{"command": "pause"}' > /tmp/narrator-ctl.json
echo '{"command": "resume"}' > /tmp/narrator-ctl.json
```

**Check status:**
```bash
cat /tmp/narrator-status.json
```

**Stop:**
```bash
tmux kill-session -t narrator
```

**Multiple commands at once:**
```bash
echo '[{"command": "style", "value": "noir"}, {"command": "profanity", "value": "high"}]' > /tmp/narrator-ctl.json
```

The `-t` / `--time` flag accepts durations like `90s`, `2m`, `5min`, `1h`.

## Live Control Protocol

The narrator polls the control file each iteration (~3s). When it finds commands, it processes them and deletes the file. Write JSON to the control file path — the narrator handles the rest.

Supported commands:

| Command | Value | Example |
|---|---|---|
| `style` | Style name | `{"command": "style", "value": "wrestling"}` |
| `profanity` | `low`, `medium`, `high` | `{"command": "profanity", "value": "low"}` |
| `pause` | (none) | `{"command": "pause"}` |
| `resume` | (none) | `{"command": "resume"}` |
| `dual_lane` | `true` / `false` | `{"command": "dual_lane", "value": true}` |

Status file returns JSON with: `style`, `profanity`, `paused`, `dual_lane`, `turn_index`, `pid`.

## Style Guide

| Style | Vibe |
|---|---|
| `sports` | Punchy play-by-play announcer, light roast |
| `nature` | David Attenborough nature documentary |
| `horror` | Creeping dread, ominous foreshadowing |
| `noir` | Hard-boiled detective, rain-soaked cynicism |
| `reality_tv` | Reality TV confessional booth commentary |
| `asmr` | Whispered meditation over mundane browsing |
| `wrestling` | BAH GAWD maximum hype announcer |

Available styles: `sports`, `nature`, `horror`, `noir`, `reality_tv`, `asmr`, `wrestling`

## Configuration

Voice IDs and ambient tracks live in `~/.narrator/config.yaml`:

```yaml
voices:
  sports: your-voice-id
  noir: your-voice-id
  wrestling: your-voice-id

ambient:
  wrestling: ~/narrator/ambient/wrestling.wav
  noir: ~/narrator/ambient/noir.wav

defaults:
  style: sports
  profanity: high
```

## Common User Requests

- "narrate my screen" / "roast my screen" -> Start with sports style
- "haunt my screen" -> Start with horror style
- "narrate for 5 minutes" -> Start with `-t 5m`
- "what styles are there" -> `python -m narrator --list`
- "switch to wrestling" -> Write style command to control file
- "make it family friendly" -> Set profanity to low
- "pause the narrator" / "shut up" -> Pause command
- "keep going" / "unpause" -> Resume command
- "stop narrating" -> Kill the tmux session or the process
