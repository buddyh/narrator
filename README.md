# Screen Narrator

Live narration of your screen activity on macOS. Captures your screen via Gemini Flash vision, generates style-specific commentary, and speaks it aloud with ElevenLabs TTS. 7 narration styles with per-style voices and ambient background tracks.

## Styles

| Style | Vibe |
|---|---|
| `sports` | Punchy play-by-play announcer |
| `nature` | David Attenborough documentary |
| `horror` | Creeping dread, ominous foreshadowing |
| `noir` | Hard-boiled detective narration |
| `reality_tv` | Reality TV confessional booth commentary |
| `asmr` | Whispered meditation |
| `wrestling` | BAH GAWD maximum hype announcer |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```bash
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=your_default_voice_id
```

Configure per-style voices and ambient tracks in `~/.narrator/config.yaml`:

```yaml
voices:
  sports: your-voice-id
  noir: your-voice-id
  wrestling: your-voice-id
  horror: your-voice-id
  asmr: your-voice-id
  reality_tv: your-voice-id

ambient:
  sports: ~/narrator/ambient/sports.wav
  noir: ~/narrator/ambient/noir.wav
  wrestling: ~/narrator/ambient/wrestling.wav
  horror: ~/narrator/ambient/horror.wav
  asmr: ~/narrator/ambient/asmr.wav
  nature: ~/narrator/ambient/nature.wav
  reality_tv: ~/narrator/ambient/reality_tv.wav

defaults:
  style: sports
  profanity: high
```

## Usage

```bash
python -m narrator                    # interactive style picker
python -m narrator horror             # specific style
python -m narrator wrestling -t 5m    # auto-stop after 5 minutes
python -m narrator --list             # show available styles
python -m narrator --dry-run          # print lines without speaking
python -m narrator --verbose          # debug output
```

## Live Control

Start with control files to change settings on the fly:

```bash
python -m narrator noir \
  --control-file /tmp/narrator-ctl.json \
  --status-file /tmp/narrator-status.json
```

Then write commands to the control file:

```bash
# Switch style
echo '{"command": "style", "value": "wrestling"}' > /tmp/narrator-ctl.json

# Change profanity level (low/medium/high)
echo '{"command": "profanity", "value": "low"}' > /tmp/narrator-ctl.json

# Pause / resume
echo '{"command": "pause"}' > /tmp/narrator-ctl.json
echo '{"command": "resume"}' > /tmp/narrator-ctl.json
```

## Architecture

```
Screen Capture --> Gemini Flash (vision + text) --> ElevenLabs TTS (WebSocket streaming)
     |                                                      |
     +-- ambient background track (per-style, looping) -----+
```

- **Dual-lane pipeline**: alternates short and long narration calls to eliminate dead air
- **Per-style voices**: each style can use a different ElevenLabs voice
- **Per-style ambient**: background audio loops with crossfade, auto-selected by style
- **Live control**: JSON control file protocol for runtime style/profanity/pause changes

## Ambient Tracks

Place 16-bit PCM mono WAV files (16kHz) in the `ambient/` directory, named by style:

```
ambient/sports.wav
ambient/noir.wav
ambient/wrestling.wav
...
```

Convert from MP3: `ffmpeg -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 ambient/style.wav`

## Notes

- macOS only (uses screen capture APIs)
- Grant Screen Recording permission to your terminal in System Settings > Privacy & Security
- Uses ElevenLabs WebSocket streaming (v2.5 models only, not v3)
- Uses Gemini Flash for vision — requires `GEMINI_API_KEY`

## Tests

```bash
python -m pytest tests/ -q
```
