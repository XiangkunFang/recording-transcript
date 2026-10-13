"""Small ONNX models, downloaded into ./models on first use."""
import urllib.request
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # corporate proxies: trust the macOS keychain

MODELS_DIR = Path(__file__).parent / "models"
URLS = {
    "silero_vad.onnx":
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "wespeaker_en_voxceleb_resnet34.onnx":
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
}


def model_path(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        print(f"Downloading {name}...")
        MODELS_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(urllib.request.urlopen(URLS[name], timeout=120).read())
        tmp.rename(path)
    return path
