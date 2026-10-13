"""Live meeting transcriber: microphone = Me, system audio (ScreenCaptureKit) = Others.

    python main.py                 # start listening
    python main.py --list-devices  # show audio devices
"""
import argparse
import queue
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # use macOS keychain certs (corporate proxies) for model download

import sounddevice as sd  # noqa: E402

from audio import Channel, MicSource, Utterance, find_input_device, take_batch  # noqa: E402
from display import COLORS, DIM, RESET, Display  # noqa: E402
from transcript import Transcript  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--mic", help="mic device name/index (default: system default input)")
    p.add_argument("--no-system", action="store_true", help="only transcribe the microphone")
    p.add_argument("--out", default="transcripts", help="folder for transcript files")
    p.add_argument("--no-speakers", action="store_true", help="label all system audio as 'Others'")
    p.add_argument("--speaker-threshold", type=float, default=None,
                   help="voice match threshold (default 0.72); lower if one person is split, raise if people merge")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    utterances: queue.Queue[Utterance | None] = queue.Queue()
    channels = [Channel("Me", MicSource(find_input_device(args.mic)), utterances)]
    if not args.no_system:
        from system_audio import SystemAudioSource
        try:
            channels.append(Channel("Others", SystemAudioSource(), utterances))
        except PermissionError as e:
            print(f"[!] {e}\n    Continuing with the microphone only.\n")

    print(f"{DIM}Loading speech model (first run downloads ~1.2GB)...{RESET}")
    from transcriber import Transcriber
    stt = Transcriber()

    tracker = None
    if not args.no_speakers and any(ch.speaker == "Others" for ch in channels):
        from speakers import DEFAULT_THRESHOLD, SpeakerTracker, WeSpeakerEmbedder
        tracker = SpeakerTracker(WeSpeakerEmbedder(), args.speaker_threshold or DEFAULT_THRESHOLD)

    for ch in list(channels):
        try:
            ch.start()
        except PermissionError as e:
            print(f"[!] {e}\n    Continuing without {ch.speaker}.\n")
            channels.remove(ch)
            continue
        print(f"{COLORS[ch.speaker]}● {ch.speaker:<6}{RESET} ← {ch.source.name}")

    sources = {ch.speaker: ch.source.name for ch in channels}
    if tracker and "Others" in sources:
        sources["Others"] += " — voices numbered Speaker 1, 2, … by timbre"
    transcript = Transcript(Path(args.out), sources)
    print(f"{DIM}Listening. Transcript → {transcript.path}   (Ctrl+C to stop){RESET}\n")
    display = Display()

    def handle(u: Utterance):
        if not u.final and not display.live_enabled:
            return
        text = stt.transcribe(u.audio)
        if not u.final:
            display.partial(u.speaker, u.started_at, text)
            return
        if not text:
            display.final(u.speaker, u.started_at, u.ended_at, "")
            return
        who = tracker.identify(u.audio) if tracker and u.speaker == "Others" else u.speaker
        display.final(u.speaker, u.started_at, u.ended_at, text, label=who)
        transcript.add(who, u.started_at, u.ended_at, text)

    try:
        while True:
            for u in take_batch(utterances):
                handle(u)
    except KeyboardInterrupt:
        display.message(f"\n{DIM}Stopping...{RESET}")
        for ch in channels:
            ch.stop()
        while not utterances.empty():
            u = utterances.get()
            if u.final:
                handle(u)
    transcript.close()
    print(f"Saved {transcript.path}")


if __name__ == "__main__":
    sys.exit(main())
