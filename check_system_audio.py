"""Self-check: play a sentence through the speakers, capture it via ScreenCaptureKit, transcribe it.

    python check_system_audio.py
"""
import subprocess
import time

import numpy as np
import truststore

truststore.inject_into_ssl()

from system_audio import SystemAudioSource  # noqa: E402
from vad import SAMPLE_RATE  # noqa: E402

SENTENCE = "The quarterly budget review is scheduled for Thursday afternoon."

chunks = []
src = SystemAudioSource()
src.start(chunks.append)
t0 = time.time()
subprocess.run(["say", "-v", "Samantha", SENTENCE], check=True)
time.sleep(0.5)
elapsed = time.time() - t0
src.stop()

audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
rate = len(audio) / elapsed
print(f"captured {len(chunks)} chunks, {len(audio)} samples in {elapsed:.1f}s ≈ {rate:.0f} Hz, peak {np.abs(audio).max(initial=0):.3f}")
if abs(rate - SAMPLE_RATE) > SAMPLE_RATE * 0.15:
    print(f"[!] expected ~{SAMPLE_RATE} Hz — sample format mismatch")
if np.abs(audio).max(initial=0) < 1e-4:
    raise SystemExit("[!] only silence captured — check volume / permission")

from transcriber import Transcriber  # noqa: E402

print("expected:", SENTENCE)
print("heard:   ", Transcriber().transcribe(audio))
