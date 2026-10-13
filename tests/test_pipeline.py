import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from segmenter import Segmenter  # noqa: E402
from vad import SAMPLE_RATE, SileroVAD  # noqa: E402

energy_vad = lambda frame: 1.0 if np.abs(frame).mean() > 0.05 else 0.0


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def feed_in_chunks(seg, audio, chunk=1024):
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.feed(audio[i:i + chunk])
    return out + seg.flush()


def test_two_utterances_split_on_silence():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(2.0), silence(1.0)])
    segs = feed_in_chunks(Segmenter(energy_vad), audio)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(0.5 - 0.2, abs=0.07)   # includes 200ms pre-roll
    assert segs[1].start == pytest.approx(2.5 - 0.2, abs=0.07)


def test_short_blip_is_dropped():
    audio = np.concatenate([silence(0.5), tone(0.15), silence(1.0)])
    assert feed_in_chunks(Segmenter(energy_vad), audio) == []


def test_long_speech_is_force_split():
    segs = feed_in_chunks(Segmenter(energy_vad, max_speech_s=5), tone(12.0))
    assert len(segs) == 3
    assert all(len(s.audio) <= 5 * SAMPLE_RATE + 512 for s in segs)


def test_flush_emits_unfinished_speech():
    seg = Segmenter(energy_vad)
    assert seg.feed(tone(1.0)) == []
    assert len(seg.flush()) == 1


def say_wav(text, path):
    subprocess.run(["say", "-v", "Samantha", "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


@pytest.mark.slow
def test_end_to_end_with_real_models(tmp_path):
    from transcriber import Transcriber

    a = say_wav("Good morning everyone, let's review the quarterly budget.", tmp_path / "a.wav")
    b = say_wav("I think we should schedule the follow up meeting for Thursday.", tmp_path / "b.wav")
    audio = np.concatenate([silence(0.5), a, silence(1.5), b, silence(1.0)])

    segs = feed_in_chunks(Segmenter(SileroVAD()), audio)
    assert len(segs) == 2

    stt = Transcriber()
    texts = [stt.transcribe(s.audio).lower() for s in segs]
    print(texts)
    assert "quarterly budget" in texts[0]
    assert "thursday" in texts[1]


def test_end_time_excludes_trailing_silence():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0)])
    (seg,) = feed_in_chunks(Segmenter(energy_vad), audio)
    assert seg.end == pytest.approx(1.5, abs=0.07)


def test_current_returns_in_progress_speech():
    seg = Segmenter(energy_vad)
    assert seg.current() is None
    seg.feed(np.concatenate([silence(0.5), tone(1.0)]))
    cur = seg.current()
    assert cur is not None and cur.start == pytest.approx(0.3, abs=0.07)
    assert len(cur.audio) >= 1.0 * SAMPLE_RATE


def test_take_batch_keeps_finals_and_latest_partial():
    import queue
    from audio import Utterance, take_batch
    q = queue.Queue()
    u = lambda sp, n, final: Utterance(sp, n, n, np.zeros(1), final)
    for item in [u("Me", 1, False), u("Me", 2, False), u("Others", 3, False), u("Me", 4, True), u("Me", 5, False)]:
        q.put(item)
    kept = [(x.speaker, x.started_at, x.final) for x in take_batch(q)]
    assert kept == [("Others", 3, False), ("Me", 4, True), ("Me", 5, False)]


def test_display_redraws_live_lines_in_place():
    import io
    from display import Display
    out = io.StringIO()
    d = Display(out, live=True)
    d.partial("Me", 0, "hello")
    d.partial("Me", 0, "hello world")
    d.final("Me", 0, 2, "Hello world.")
    s = out.getvalue()
    assert s.count("\033[1F\033[J") == 2     # each update erased the previous live line
    assert s.rstrip().endswith("Hello world.")


def test_markdown_transcript(tmp_path):
    from transcript import Transcript
    t = Transcript(tmp_path, {"Me": "Mic", "Others": "System audio"})
    t.add("Me", 0, 3, "Hello there.")
    t.close()
    md = t.path.read_text()
    assert t.path.suffix == ".md"
    assert md.startswith("# Meeting transcript")
    assert "- **Others**: System audio" in md
    assert "] Me:** Hello there.\n\n" in md
    assert "*Ended " in md


def test_speaker_tracker_groups_by_voiceprint():
    from speakers import SpeakerTracker
    voices = {"a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0])}
    noisy = lambda v: lambda audio: voices[v] + 0.1 * np.ones(3)
    t = SpeakerTracker(embed=None)
    labels = []
    for v in "abab":
        t.embed = noisy(v)
        labels.append(t.identify(np.zeros(SAMPLE_RATE * 2, np.float32)))
    assert labels == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"]


def test_short_utterance_keeps_last_speaker():
    from speakers import SpeakerTracker
    t = SpeakerTracker(embed=lambda a: np.array([1.0, 0]))
    assert t.identify(np.zeros(int(SAMPLE_RATE * 0.5), np.float32)) == "Speaker 1"   # nothing known yet
    t.identify(np.zeros(SAMPLE_RATE * 2, np.float32))
    t.embed = lambda a: np.array([0, 1.0])
    t.identify(np.zeros(SAMPLE_RATE * 2, np.float32))                                  # Speaker 2 talks
    t.embed = lambda a: pytest.fail("short clips must not be embedded")
    assert t.identify(np.zeros(int(SAMPLE_RATE * 0.5), np.float32)) == "Speaker 2"


@pytest.mark.slow
def test_real_voices_are_told_apart():
    """3 real LibriSpeech speakers (CC BY 4.0), 3 sentences each, interleaved like a meeting."""
    from speakers import SpeakerTracker, WeSpeakerEmbedder
    d = np.load(Path(__file__).parent / "fixtures" / "librispeech_3spk.npz")
    spk = d["spk"]
    order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    t = SpeakerTracker(WeSpeakerEmbedder())
    got = [t.identify(d[f"a{i}"].astype(np.float32) / 32768) for i in order]
    truth = [int(spk[i]) for i in order]
    mapping = dict(zip(truth, got))
    assert len(set(got)) == 3
    assert got == [mapping[s] for s in truth], list(zip(truth, got))
