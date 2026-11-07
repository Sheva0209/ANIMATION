"""
audio_check.py
Show where the silent gaps are in scene audio (to find the "dead air" you hear in videos).

Usage:
  python src/audio_check.py audio/cache/<scene>.wav [more.wav ...]
  python src/audio_check.py            (no args: checks the 8 newest scene WAVs in audio/cache)
"""
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence

BASE_DIR = Path(__file__).parent.parent
OFFSET_DB = 16      # same rule as tts_generator: silence = 16 dB below the scene's average loudness
MIN_GAP_MS = 400    # only report gaps at least this long


def check(path: Path) -> None:
    seg = AudioSegment.from_file(str(path))
    total = len(seg)
    if seg.dBFS == float("-inf"):
        print(f"{path.name}: {total / 1000:.1f}s  (completely silent!)")
        return
    gaps = detect_silence(seg, min_silence_len=MIN_GAP_MS, silence_thresh=seg.dBFS - OFFSET_DB, seek_step=10)
    dead = sum(e - s for s, e in gaps)
    print(f"\n{path.name}: {total / 1000:.1f}s total | silence >= {MIN_GAP_MS}ms: {dead / 1000:.1f}s "
          f"({100 * dead // max(total, 1)}%) in {len(gaps)} gap(s)")
    for s, e in gaps:
        where = "START" if s <= 50 else ("END" if e >= total - 50 else "middle")
        print(f"   {s / 1000:6.2f}s -> {e / 1000:6.2f}s   {(e - s) / 1000:5.2f}s  [{where}]")


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        cache = BASE_DIR / "audio" / "cache"
        wavs = [p for p in cache.glob("*.wav") if ".t" not in p.stem and ".tmp" not in p.stem]
        files = sorted(wavs, key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if not files:
        print("No WAV files found.")
    for f in files:
        check(f)