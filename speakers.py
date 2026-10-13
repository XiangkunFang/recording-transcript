"""Tells voices apart by timbre: online clustering of speaker embeddings ("Speaker 1", "Speaker 2", ...)."""
from typing import Callable

import numpy as np

from vad import SAMPLE_RATE

# Calibrated on real LibriSpeech speakers: ~96% of sentences right for 2-3 speakers.
# Lower it if one person gets split into several; raise it if different people get merged.
DEFAULT_THRESHOLD = 0.72


class WeSpeakerEmbedder:
    """256-dim voice fingerprint (WeSpeaker ResNet34, VoxCeleb), ~30ms per sentence."""

    def __init__(self):
        import sherpa_onnx

        from models import model_path
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path("wespeaker_en_voxceleb_resnet34.onnx")), num_threads=2
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        return np.array(self._extractor.compute(stream))


class SpeakerTracker:
    def __init__(
        self,
        embed: Callable[[np.ndarray], np.ndarray],
        threshold: float = DEFAULT_THRESHOLD,
        min_seconds: float = 1.0,
    ):
        self.embed = embed
        self.threshold = threshold
        self.min_samples = int(min_seconds * SAMPLE_RATE)
        self._sums: list[np.ndarray] = []   # per speaker: sum of unit embeddings (direction = centroid)
        self._last: int | None = None

    def identify(self, audio: np.ndarray) -> str:
        if len(audio) < self.min_samples:
            # too short for a reliable voiceprint ("yeah", "ok") — assume the last speaker continued
            return self._name(self._last or 0)

        x = self.embed(audio)
        x = x / (np.linalg.norm(x) + 1e-9)
        sims = [x @ s / np.linalg.norm(s) for s in self._sums]
        if sims and max(sims) >= self.threshold:
            k = int(np.argmax(sims))
            self._sums[k] = self._sums[k] + x
        else:
            k = len(self._sums)
            self._sums.append(x)
        self._last = k
        return self._name(k)

    @staticmethod
    def _name(k: int) -> str:
        return f"Speaker {k + 1}"
