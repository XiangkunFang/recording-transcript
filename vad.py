"""Silero VAD v5 (ONNX) — returns speech probability for 512-sample frames at 16kHz."""
import numpy as np
import onnxruntime as ort

from models import model_path

SAMPLE_RATE = 16000
FRAME = 512      # samples per VAD frame (32ms)
_CONTEXT = 64    # Silero v5 expects the last 64 samples of the previous frame prepended


class SileroVAD:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path("silero_vad.onnx")), sess_options=opts)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.reshape(1, FRAME).astype(np.float32)], axis=1)
        prob, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -_CONTEXT:]
        return float(prob[0, 0])
