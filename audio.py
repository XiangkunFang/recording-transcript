"""Audio sources and per-speaker channels that turn a source into utterances."""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd

from segmenter import Segmenter
from vad import SAMPLE_RATE, SileroVAD

PARTIAL_EVERY_S = 0.4   # seconds of audio between live-caption updates


@dataclass
class Utterance:
    speaker: str
    started_at: float   # unix time
    ended_at: float     # unix time
    audio: np.ndarray
    final: bool = True  # False = in-progress snapshot for live captions


class Source(Protocol):
    """Delivers 16kHz mono float32 chunks to `push` from any thread."""
    name: str
    def start(self, push: Callable[[np.ndarray], None]) -> None: ...
    def stop(self) -> None: ...


def find_input_device(name_part: str | None) -> int | None:
    """Index of the first input device whose name contains name_part (case-insensitive)."""
    if name_part is None:
        return None
    if name_part.isdigit():
        return int(name_part)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_part.lower() in d["name"].lower():
            return i
    raise SystemExit(f"No input device matching {name_part!r}. Try --list-devices.")


class MicSource:
    def __init__(self, device: int | None = None):
        info = sd.query_devices(device, "input")
        self.name = info["name"]
        self._device = device
        self._channels = min(2, info["max_input_channels"])
        self._stream = None

    def start(self, push):
        self._stream = sd.InputStream(
            device=self._device, samplerate=SAMPLE_RATE, channels=self._channels,
            dtype="float32", blocksize=1024,
            callback=lambda indata, *_: push(indata.mean(axis=1).copy()),
        )
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()


class Channel:
    """One speaker = one source. Source -> raw queue -> segmenter thread -> utterances."""

    def __init__(self, speaker: str, source: Source, out: "queue.Queue[Utterance]"):
        self.speaker = speaker
        self.source = source
        self.out = out
        self._raw: queue.Queue[np.ndarray | None] = queue.Queue()
        self._segmenter = Segmenter(SileroVAD())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = 0.0
        self._since_partial = 0

    def start(self):
        self._t0 = time.time()
        self._thread.start()
        self.source.start(self._raw.put)

    def stop(self):
        self.source.stop()
        self._raw.put(None)
        self._thread.join()

    def _run(self):
        while (chunk := self._raw.get()) is not None:
            for seg in self._segmenter.feed(chunk):
                self._emit(seg)
                self._since_partial = 0
            self._since_partial += len(chunk)
            if self._since_partial >= PARTIAL_EVERY_S * SAMPLE_RATE:
                if (cur := self._segmenter.current()) is not None:
                    self._emit(cur, final=False)
                    self._since_partial = 0
        for seg in self._segmenter.flush():
            self._emit(seg)

    def _emit(self, seg, final=True):
        self.out.put(Utterance(self.speaker, self._t0 + seg.start, self._t0 + seg.end, seg.audio, final))


def take_batch(q: "queue.Queue[Utterance]") -> list[Utterance]:
    """Block for one item, drain the rest; drop partials superseded by a later item of the same speaker."""
    batch = [q.get()]
    while True:
        try:
            batch.append(q.get_nowait())
        except queue.Empty:
            break
    return [
        u for i, u in enumerate(batch)
        if u.final or not any(v.speaker == u.speaker for v in batch[i + 1:])
    ]
