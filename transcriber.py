"""English speech-to-text with NVIDIA Parakeet running on Apple Silicon via MLX."""
import mlx.core as mx
import numpy as np
from huggingface_hub.utils import logging as hf_logging
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

hf_logging.set_verbosity_error()

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model = from_pretrained(model_id)
        self.cfg = self.model.preprocessor_config
        self.transcribe(np.zeros(self.cfg.sample_rate, dtype=np.float32))  # warm up

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < self.cfg.hop_length:
            return ""
        mel = get_logmel(mx.array(audio, dtype=mx.float32), self.cfg)
        return self.model.generate(mel)[0].text.strip()
