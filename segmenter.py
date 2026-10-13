"""Turns a continuous 16kHz stream into utterances using a VAD callable."""
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vad import FRAME, SAMPLE_RATE


@dataclass
class Segment:
    start: float        # seconds since the stream started
    end: float          # end of speech (trailing silence excluded)
    audio: np.ndarray   # float32 mono, 16kHz


class Segmenter:
    def __init__(
        self,
        vad: Callable[[np.ndarray], float],
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        min_speech_ms: int = 400,
        max_speech_s: float = 15.0,
        pad_ms: int = 200,
    ):
        self.vad = vad
        self.on = threshold
        self.off = max(threshold - 0.15, 0.01)
        ms_to_frames = lambda ms: max(1, int(ms * SAMPLE_RATE / 1000 / FRAME))
        self.min_silence = ms_to_frames(min_silence_ms)
        self.min_speech = ms_to_frames(min_speech_ms)
        self.max_frames = int(max_speech_s * SAMPLE_RATE / FRAME)
        self.preroll = deque(maxlen=ms_to_frames(pad_ms))

        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_idx = 0      # index of the next frame to be processed
        self._speaking = False
        self._buf: list[np.ndarray] = []
        self._start_idx = 0
        self._voiced = 0
        self._silence = 0

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Feed any number of samples; returns utterances that finished."""
        self._pending = np.concatenate([self._pending, samples.astype(np.float32)])
        out = []
        n = len(self._pending) // FRAME
        for i in range(n):
            seg = self._step(self._pending[i * FRAME:(i + 1) * FRAME])
            if seg is not None:
                out.append(seg)
        self._pending = self._pending[n * FRAME:]
        return out

    def current(self) -> Segment | None:
        """The utterance still in progress (for live captions), or None."""
        if not self._speaking or self._voiced < self.min_speech:
            return None
        return Segment(
            start=self._start_idx * FRAME / SAMPLE_RATE,
            end=(self._start_idx + len(self._buf)) * FRAME / SAMPLE_RATE,
            audio=np.concatenate(self._buf),
        )

    def flush(self) -> list[Segment]:
        seg = self._emit() if self._speaking else None
        return [seg] if seg is not None else []

    def _step(self, frame: np.ndarray) -> Segment | None:
        prob = self.vad(frame)
        idx = self._frame_idx
        self._frame_idx += 1

        if not self._speaking:
            if prob >= self.on:
                self._speaking = True
                self._buf = list(self.preroll) + [frame]
                self._start_idx = idx - len(self.preroll)
                self._voiced, self._silence = 1, 0
                self.preroll.clear()
            else:
                self.preroll.append(frame)
            return None

        self._buf.append(frame)
        if prob < self.off:
            self._silence += 1
        else:
            self._silence = 0
            self._voiced += 1

        if self._silence >= self.min_silence:
            return self._emit()
        if len(self._buf) >= self.max_frames:
            seg = self._emit()
            self._speaking = True   # still talking — keep collecting into a new segment
            self._start_idx = self._frame_idx
            return seg
        return None

    def _emit(self) -> Segment | None:
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        voiced = self._voiced
        start = self._start_idx * FRAME / SAMPLE_RATE
        end = (self._start_idx + len(self._buf) - self._silence) * FRAME / SAMPLE_RATE
        self._speaking = False
        self._buf, self._voiced, self._silence = [], 0, 0
        if voiced < self.min_speech:
            return None
        return Segment(start=start, end=end, audio=audio)
