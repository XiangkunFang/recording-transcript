"""Markdown transcript file, written line by line so nothing is lost on a crash."""
from datetime import datetime
from pathlib import Path


def _hms(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


class Transcript:
    def __init__(self, folder: Path, sources: dict[str, str]):
        folder.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        self.path = folder / f"{now:%Y-%m-%d_%H%M%S}.md"
        self._f = self.path.open("w", encoding="utf-8")
        self._write(f"# Meeting transcript — {now:%Y-%m-%d %H:%M}\n\n")
        for speaker, name in sources.items():
            self._write(f"- **{speaker}**: {name}\n")
        self._write("\n---\n\n")

    def add(self, speaker: str, started_at: float, ended_at: float, text: str):
        self._write(f"**[{_hms(started_at)} – {_hms(ended_at)}] {speaker}:** {text}\n\n")

    def close(self):
        self._write(f"---\n\n*Ended {datetime.now():%H:%M:%S}*\n")
        self._f.close()

    def _write(self, s: str):
        self._f.write(s)
        self._f.flush()
