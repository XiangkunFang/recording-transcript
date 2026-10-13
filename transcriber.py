"""English speech-to-text with NVIDIA Parakeet running on Apple Silicon via MLX."""
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import logging as hf_logging
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

hf_logging.set_verbosity_error()

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


def _cached_dir(model_id: str) -> str | None:
    """Local snapshot folder if the model is fully downloaded (checked on disk, no network)."""
    paths = [try_to_load_from_cache(model_id, f) for f in ("config.json", "model.safetensors")]
    if all(isinstance(p, str) for p in paths):
        return str(Path(paths[0]).parent)
    return None


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        # Load from cache without asking the Hub: avoids a long timeout when a proxy blocks huggingface.co
        self.model = from_pretrained(_cached_dir(model_id) or model_id)
        self.cfg = self.model.preprocessor_config
        self.transcribe(np.zeros(self.cfg.sample_rate, dtype=np.float32))  # warm up

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < self.cfg.hop_length:
            return ""
        mel = get_logmel(mx.array(audio, dtype=mx.float32), self.cfg)
        return self.model.generate(mel)[0].text.strip()
