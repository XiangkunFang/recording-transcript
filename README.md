# Live meeting transcriber

Real-time English transcription of Teams/Zoom calls, labelled **Me** (your microphone) vs **Speaker 1 / 2 / 3…** (people on the call, told apart by voice; captured from system audio via macOS ScreenCaptureKit — no virtual audio driver needed). Runs fully on-device: NVIDIA Parakeet TDT 0.6B v2 on Apple Silicon (MLX), Silero VAD, WeSpeaker ResNet34 voiceprints. No audio leaves the machine.

Words appear live while someone is talking (grey line at the bottom, refreshed every ~0.4 s); when the sentence ends it is replaced by the final, coloured line:

```
[11:02:13 - 11:02:16] Me: Good morning everyone, let's review the quarterly budget.
[11:02:18 - 11:02:20] Speaker 1: Sure, I'll share my screen.
[11:02:21] Me: So the first item on the agenda is      ← live, still talking
```

Transcripts are saved as Markdown to `transcripts/YYYY-MM-DD_HHMMSS.md` (final sentences only, with start and end time; written line by line, so a crash loses nothing):

```markdown
# Meeting transcript — 2026-10-12 11:02

- **Me**: MacBook Pro Microphone
- **Others**: System audio (ScreenCaptureKit) — voices numbered Speaker 1, 2, … by timbre

---

**[11:02:13 – 11:02:16] Me:** Good morning everyone, let's review the quarterly budget.

**[11:02:18 – 11:02:20] Speaker 1:** Sure, I'll share my screen.
```

Audio itself is not saved.

## One-time setup

1. Python deps:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
2. Permissions for the app you run Python from (Terminal / iTerm / VS Code), in System Settings → Privacy & Security:
   - **Microphone** → enable it (for "Me")
   - **Screen & System Audio Recording** → enable it (for "Others"), then restart that app
3. Verify system-audio capture (plays a test sentence and transcribes it):
   ```bash
   .venv/bin/python check_system_audio.py
   ```

## Run

```bash
.venv/bin/python main.py
```

Options: `--list-devices`, `--mic "MacBook Pro"` (pick a mic by name or index), `--no-system` (mic only), `--no-speakers` (label everyone on the call "Others"), `--speaker-threshold 0.72`, `--out <folder>`. Ctrl+C to stop.

## Notes

- **Wear headphones.** On speakers the mic hears the other side too, and their speech shows up again as "Me".
- Live words lag speech by ~0.5 s and may self-correct; the final line (and the file) uses the whole sentence, so it's the most accurate. A sentence closes after 600 ms of silence; long monologues are split every 15 s.
- AirPods' mic forces Bluetooth into low-quality call mode; the built-in MacBook mic (`--mic "MacBook Pro"`) usually transcribes better.
- **Telling people apart**: each finished sentence gets a voiceprint and is matched to the voices heard so far; a new voice becomes the next `Speaker N`. Numbers restart every meeting (no names — find-and-replace in the .md afterwards). Tested on real LibriSpeech voices: ~96% of sentences correct with 2–3 people, ~94% with 5. Expect less on compressed call audio or similar-sounding voices. If one person keeps getting split into two numbers, lower `--speaker-threshold` (e.g. 0.65); if two people get merged, raise it (e.g. 0.78). Sentences under 1 s ("yeah", "ok") are given to whoever spoke last; a quick hand-over with no pause stays one sentence with one label. Live grey lines say "Others" until the sentence is final.
- Let the other participants know you are transcribing.

## Tests

```bash
.venv/bin/python -m pytest tests -v
```

`tests/fixtures/librispeech_3spk.npz` contains 9 short clips from [LibriSpeech](https://www.openslr.org/12) dev-clean (CC BY 4.0), used to test speaker separation on real voices.
