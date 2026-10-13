"""Terminal output: finished lines scroll up, live captions redraw in place at the bottom."""
import shutil
import sys
import time

COLORS = {"Me": "\033[96m", "Others": "\033[93m"}
SPEAKER_COLORS = ["\033[93m", "\033[95m", "\033[92m", "\033[91m", "\033[94m", "\033[33m"]
RESET, DIM = "\033[0m", "\033[2m"


def color(speaker: str) -> str:
    if speaker in COLORS:
        return COLORS[speaker]
    n = int(speaker.rsplit(" ", 1)[-1]) if speaker[-1].isdigit() else 1   # "Speaker 3" -> 3
    return SPEAKER_COLORS[(n - 1) % len(SPEAKER_COLORS)]


def _stamp(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


class Display:
    def __init__(self, out=sys.stdout, live: bool | None = None):
        self.out = out
        self.live_enabled = out.isatty() if live is None else live
        self._live: dict[str, str] = {}   # speaker -> rendered live line (no colors)
        self._drawn = 0                   # live lines currently on screen

    def partial(self, speaker: str, started_at: float, text: str):
        if not self.live_enabled or not text:
            return
        self._live[speaker] = f"[{_stamp(started_at)}] {speaker}: {text}"
        self._redraw()

    def final(self, speaker: str, started_at: float, ended_at: float, text: str, label: str | None = None):
        """speaker = channel whose live line this replaces; label = who to show (e.g. "Speaker 2")."""
        self._live.pop(speaker, None)
        self._clear()
        if text:
            who = label or speaker
            when = f"[{_stamp(started_at)} - {_stamp(ended_at)}]"
            self.out.write(f"{DIM}{when}{RESET} {color(who)}{who}:{RESET} {text}\n")
        self._draw()

    def message(self, text: str):
        self._clear()
        self.out.write(text + "\n")
        self._draw()

    def _redraw(self):
        self._clear()
        self._draw()

    def _clear(self):
        if self._drawn:
            self.out.write(f"\033[{self._drawn}F\033[J")  # up N lines to column 0, erase below
            self._drawn = 0

    def _draw(self):
        width = shutil.get_terminal_size().columns - 1
        for line in self._live.values():
            if len(line) > width:           # never wrap, or the cursor math breaks
                line = "…" + line[-(width - 1):]
            self.out.write(f"{DIM}{line}{RESET}\n")
        self._drawn = len(self._live)
        self.out.flush()
