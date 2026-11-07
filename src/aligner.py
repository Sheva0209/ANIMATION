"""
aligner.py
Word-level subtitle timing with faster-whisper (local, free, no API).

How it works
  1. Each scene's cached WAV is transcribed by Whisper with word timestamps
     (the script narration is passed as initial_prompt so spelling follows the script).
  2. Whisper's words are matched to the SCRIPT words (difflib), so the subtitle text is
     always exactly the script -- only the timing comes from Whisper. Words Whisper
     missed or mangled get interpolated timings between their neighbours.
  3. Result is cached next to the scene WAV (audio/cache/<hash>.words.<model>.json),
     so an unchanged scene is never transcribed twice.

Model cache location
  The Whisper model is downloaded into the Hugging Face hub cache that the rest of the
  project already uses (HF_HOME from .env, e.g. D:/hf_cache). If that cache resolves to
  the C: drive, this module REFUSES to download and tells you how to fix it.

Quick test (no pipeline needed):
  python src/aligner.py path/to/scene.wav "the narration text of that audio"
"""

import hashlib
import json
import os
import re
import difflib
from pathlib import Path, PureWindowsPath

BASE_DIR = Path(__file__).parent.parent


# ─── Model cache location (must NOT be on C:) ────────────────────────────────

def _hf_hub_dir() -> Path:
    """Resolve the HF hub cache dir the same way tts_generator.py does (.env first)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except Exception:
        pass
    hf_home = Path(os.environ.get("HF_HOME", str(BASE_DIR / "TTS" / "hf-cache"))).expanduser()
    if not hf_home.is_absolute():
        hf_home = BASE_DIR / hf_home
    os.environ.setdefault("HF_HOME", str(hf_home))
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    if not hub.is_absolute():
        hub = BASE_DIR / hub
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))
    return hub


def check_cache_not_on_c(path: Path | str) -> None:
    """Raise if `path` is on the C: drive (Windows). Pure function so it is easy to test."""
    drive = PureWindowsPath(str(path)).drive.upper()
    if drive == "C:":
        raise RuntimeError(
            f"Whisper model cache would be on C: ({path}). "
            "Set HF_HOME=D:/hf_cache (and HUGGINGFACE_HUB_CACHE=D:/hf_cache/hub) in .env, "
            "then run again. Nothing was downloaded."
        )


# ─── Whisper model (lazy singleton) ──────────────────────────────────────────

_MODEL = None
_MODEL_KEY = None


def _wcfg(config: dict) -> dict:
    w = dict(config.get("whisper_local", {}) or {})
    w.setdefault("model", "base")
    w.setdefault("device", "cpu")
    w.setdefault("compute_type", "int8")
    w.setdefault("beam_size", 5)
    if not w.get("language"):
        w["language"] = (config.get("llm", {}) or {}).get("language") or None
    return w


def _get_model(wcfg: dict):
    """Load (once) a faster-whisper model; cache dir is verified to be off C: first."""
    global _MODEL, _MODEL_KEY
    key = (wcfg["model"], wcfg["device"], wcfg["compute_type"])
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL

    hub = _hf_hub_dir()
    check_cache_not_on_c(hub)
    hub.mkdir(parents=True, exist_ok=True)

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Install it inside the project's venv with:\n"
            '  pip install faster-whisper --cache-dir "D:\\pip_cache"'
        ) from e

    device, compute = wcfg["device"], wcfg["compute_type"]
    print(f"[Align] Loading Whisper '{wcfg['model']}' ({device}/{compute}) | model cache -> {hub}")
    try:
        try:
            _MODEL = WhisperModel(
                wcfg["model"], device=device, compute_type=compute, download_root=str(hub)
            )
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[Align] {device} unavailable ({e}); falling back to cpu/int8")
            _MODEL = WhisperModel(
                wcfg["model"], device="cpu", compute_type="int8", download_root=str(hub)
            )
    except Exception as e:
        # One clear, environment-level error (the caller stops retrying for the other scenes)
        raise RuntimeError(f"could not load Whisper model '{wcfg['model']}': {e}") from e
    _MODEL_KEY = key
    return _MODEL


def _transcribe_words(audio_path: str, narration: str, wcfg: dict) -> list[dict]:
    """Run Whisper -> [{'word', 'start_ms', 'end_ms'}] (relative to the start of audio_path)."""
    global _MODEL, _MODEL_KEY

    def run(cfg_):
        model = _get_model(cfg_)
        segments, _info = model.transcribe(
            str(audio_path),
            language=cfg_.get("language"),
            word_timestamps=True,
            initial_prompt=narration or None,    # biases spelling/vocabulary toward the script
            condition_on_previous_text=False,
            vad_filter=False,
            beam_size=int(cfg_["beam_size"]),
        )
        return list(segments)                    # decoding happens lazily while iterating

    try:
        segs = run(wcfg)
    except Exception as e:
        if wcfg["device"] == "cpu":
            raise
        # e.g. missing cuBLAS/cuDNN DLLs only show up at the first decode -> retry on CPU
        print(f"[Align] {wcfg['device']} failed ({e}); retrying on cpu/int8")
        _MODEL, _MODEL_KEY = None, None
        segs = run({**wcfg, "device": "cpu", "compute_type": "int8"})

    words = []
    for seg in segs:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append({
                    "word": text,
                    "start_ms": int(round(w.start * 1000)),
                    "end_ms": int(round(w.end * 1000)),
                })
    return words


# ─── Script <-> Whisper alignment (pure logic, unit-testable) ────────────────

def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower()).replace("’", "'")


def align_script_words(script_words: list[str], asr_words: list[dict], duration_ms: int) -> list[dict]:
    """
    Give every SCRIPT word a start/end time using Whisper's words as anchors.
    Returns [{'word', 'start_ms', 'end_ms'}] for exactly the script words, in order.
    """
    n = len(script_words)
    if n == 0:
        return []

    def spread(i0: int, i1: int, t0: float, t1: float, out: list):
        """Distribute script words [i0, i1) over the window [t0, t1] by character weight."""
        if i1 <= i0:
            return
        t1 = max(t1, t0 + 60 * (i1 - i0))              # never collapse to zero length
        weights = [len(script_words[i]) + 1 for i in range(i0, i1)]
        total = float(sum(weights))
        cur = t0
        for k, i in enumerate(range(i0, i1)):
            span = (t1 - t0) * weights[k] / total
            out[i] = (cur, cur + span)
            cur += span

    out: list = [None] * n

    if not asr_words:
        spread(0, n, 0, duration_ms, out)
    else:
        a = [_norm(w) for w in script_words]
        b = [_norm(w["word"]) for w in asr_words]
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        matched = 0
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                w = asr_words[blk.b + k]
                out[blk.a + k] = (w["start_ms"], w["end_ms"])
                matched += 1

        speech_start = asr_words[0]["start_ms"]
        speech_end = asr_words[-1]["end_ms"]

        if matched < max(1, int(0.3 * n)):
            # Too little agreement to trust word-by-word anchors: spread over the speech span
            out = [None] * n
            spread(0, n, speech_start, speech_end, out)
        else:
            # Interpolate the runs of unmatched script words between anchors
            i = 0
            while i < n:
                if out[i] is not None:
                    i += 1
                    continue
                j = i
                while j < n and out[j] is None:
                    j += 1
                t0 = out[i - 1][1] if i > 0 else speech_start
                t1 = out[j][0] if j < n else speech_end
                spread(i, j, t0, t1, out)
                i = j

    # Clean up: integers, ordered, inside [0, duration]
    result = []
    prev_start = 0
    for i, w in enumerate(script_words):
        s, e = out[i]
        s = int(max(prev_start, min(s, duration_ms)))
        e = int(max(s + 1, min(e, max(duration_ms, s + 1))))
        result.append({"word": w, "start_ms": s, "end_ms": e})
        prev_start = s
    return result


# ─── Per-scene entry point (with cache) ──────────────────────────────────────

def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def get_scene_word_timings(wav_path: str, narration: str, duration_ms: int, config: dict) -> list[dict]:
    """
    Word timings for one scene, relative to the scene start.
    Cached beside the WAV; the cache is invalidated if the narration, model or audio changes.
    """
    wcfg = _wcfg(config)
    wav = Path(wav_path)
    if not wav.exists():
        raise FileNotFoundError(f"scene audio not found: {wav}")

    cache_path = wav.with_name(f"{wav.stem}.words.{wcfg['model']}.json")
    meta = {"narration": _sha(narration), "size": wav.stat().st_size, "model": wcfg["model"]}

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("meta") == meta:
                return cached["words"]
        except Exception:
            pass

    asr = _transcribe_words(str(wav), narration, wcfg)
    words = align_script_words(narration.split(), asr, duration_ms)

    tmp = cache_path.with_name(cache_path.name + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "words": words}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache_path)
    return words


if __name__ == "__main__":
    import sys
    import yaml

    if len(sys.argv) < 2:
        print('Usage: python src/aligner.py <scene.wav> ["narration text"]')
        print('  without text : just transcribes the audio and prints Whisper word times (model test)')
        print('  with text    : aligns the given script text to the audio')
        raise SystemExit(1)

    cfg = yaml.safe_load(open(BASE_DIR / "config.yaml", encoding="utf-8"))
    wav = sys.argv[1]

    if len(sys.argv) == 2:
        words = _transcribe_words(wav, "", _wcfg(cfg))
        print(f"\n[Align] Whisper heard {len(words)} words:")
        print(" ".join(w["word"] for w in words))
        print()
        for w in words:
            print(f"{w['start_ms']:>6} -> {w['end_ms']:>6}  {w['word']}")
    else:
        from pydub import AudioSegment
        dur = len(AudioSegment.from_file(wav))
        for w in get_scene_word_timings(wav, sys.argv[2], dur, cfg):
            print(f"{w['start_ms']:>6} -> {w['end_ms']:>6}  {w['word']}")