"""
transcriber.py
Generate .srt subtitle from LLM script + TTS timing data.

Two timing modes (config: subtitle.word_sync):
  * word_sync: true  -> real word timings from local faster-whisper (see aligner.py),
                        chunks follow sentences/punctuation, karaoke highlight possible.
                        Also writes <name>.words.json next to the .srt.
  * word_sync: false -> old behaviour: words spread proportionally over each scene's audio.
If word sync fails for any reason (package missing, cache on C:, ...) it falls back to the
proportional mode automatically, per scene where possible.
"""

import json
import math
import os
import re
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Primary: Generate SRT directly from script + TTS audio_data (no API call)
# ---------------------------------------------------------------------------

def generate_srt_from_script(
    script: dict,
    audio_data: dict,
    config: dict | None = None,
    safe_name: str | None = None,
) -> str:
    """
    Build the subtitle .srt (+ optional .words.json) from the LLM narration + TTS timing.
    Uses real word timings from faster-whisper when subtitle.word_sync is true.
    """
    if config is None:
        config = load_config()

    sub_cfg = config["subtitle"]
    max_chars = sub_cfg["max_chars_per_line"]
    max_words = sub_cfg.get("max_words_per_chunk", 7)
    hold_ms = int(sub_cfg.get("hold_ms", 600))

    scene_audio = audio_data["scene_audio"]
    scene_offsets = audio_data["scene_offsets"]

    if safe_name is None:
        safe_name = (
            re.sub(r"[^\w\s-]", "", script.get("title", "output"))
            .strip().replace(" ", "_").lower()[:60]
        )

    srt_path = BASE_DIR / "subtitles" / f"{safe_name}.srt"
    words_path = srt_path.with_suffix(".words.json")
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    synced = _align_scenes(scene_audio, config) if sub_cfg.get("word_sync", True) else [None] * len(scene_audio)

    entries = []
    for scene_info, start_ms, words in zip(scene_audio, scene_offsets, synced):
        text = scene_info["narration"].strip()
        total_ms = scene_info["duration_ms"]

        if words:
            entries.extend(_chunks_from_words(words, start_ms, start_ms + total_ms, max_words, max_chars, hold_ms))
        else:
            for chunk in _split_into_chunks(text, total_ms, max_words):
                chunk_start = start_ms + chunk["offset_ms"]
                entries.append({
                    "start_ms": chunk_start,
                    "end_ms": chunk_start + chunk["duration_ms"],
                    "text": _wrap_text(chunk["text"], max_chars),
                    "words": None,
                })

    srt_blocks = []
    for i, e in enumerate(entries, 1):
        e["index"] = i
        srt_blocks.append(
            f"{i}\n{_ms_to_srt_time(e['start_ms'])} --> {_ms_to_srt_time(e['end_ms'])}\n{e['text']}\n"
        )
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    n_synced = sum(1 for w in synced if w)
    if n_synced:
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
    elif words_path.exists():
        words_path.unlink()   # never leave a stale word file next to a plain SRT

    mode = f"word-synced {n_synced}/{len(synced)} scenes" if n_synced else "proportional timing"
    print(f"[Subtitle] OK {mode} -> {srt_path} ({len(entries)} entries)")
    return str(srt_path)


def _align_scenes(scene_audio: list[dict], config: dict) -> list:
    """Per-scene word timings from Whisper. A failed scene -> None (falls back to proportional)."""
    try:
        import aligner
    except Exception as e:  # pragma: no cover
        print(f"[Subtitle] Word sync unavailable ({e}); using proportional timing")
        return [None] * len(scene_audio)

    results = []
    for i, info in enumerate(scene_audio):
        try:
            results.append(aligner.get_scene_word_timings(
                info["path"], info["narration"].strip(), info["duration_ms"], config
            ))
        except (RuntimeError, ImportError) as e:
            # Environment-level problem (package missing, cache on C:, no model): stop trying.
            print(f"[Subtitle] Word sync disabled: {e}")
            results.extend([None] * (len(scene_audio) - i))
            break
        except Exception as e:
            print(f"[Subtitle] Scene {info.get('scene_id', i + 1)}: word sync failed ({e}); proportional timing")
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Word-synced chunking
# ---------------------------------------------------------------------------

_SENT_END = re.compile(r"[.!?\u2026][\"')\]\u201d\u2019]*$")


def _fits_two_lines(words: list[str], max_chars: int) -> bool:
    """Would these words wrap into at most 2 lines of max_chars (same greedy rule as _wrap_text)?"""
    lines, cur = 1, ""
    for w in words:
        test = f"{cur} {w}".strip()
        if len(test) <= max_chars:
            cur = test
        else:
            lines += 1
            cur = w
    return lines <= 2


def _group_words(gw: list[dict], max_words: int, max_chars: int) -> list[list[dict]]:
    """Group words into subtitle chunks: never cross a sentence end, balanced sizes, fit 2 lines."""
    sentences, cur = [], []
    for w in gw:
        cur.append(w)
        if _SENT_END.search(w["w"]):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    def split_even(sent: list[dict], mw: int) -> list[list[dict]]:
        k = math.ceil(len(sent) / mw)
        base, rem = divmod(len(sent), k)
        parts, i = [], 0
        for p in range(k):
            size = base + (1 if p < rem else 0)
            parts.append(sent[i:i + size])
            i += size
        return parts

    groups = []
    for sent in sentences:
        chosen = None
        # 1st choice: balanced chunks that each fit on ONE line (no orphan words on line 2)
        for mw in range(max(1, max_words), 2, -1):
            parts = split_even(sent, mw)
            if all(len(" ".join(x["w"] for x in part)) <= max_chars // 1 for part in parts):
                chosen = parts
                break
        # 2nd choice: allow two lines, as long as nothing would be truncated
        if chosen is None:
            for mw in range(max(1, max_words), 0, -1):
                parts = split_even(sent, mw)
                if mw == 1 or all(_fits_two_lines([x["w"] for x in part], max_chars) for part in parts):
                    chosen = parts
                    break
        groups.extend(chosen)
    return groups


def _chunks_from_words(words: list[dict], scene_start: int, scene_end: int,
                       max_words: int, max_chars: int, hold_ms: int) -> list[dict]:
    """Turn one scene's word timings into subtitle entries (global milliseconds)."""
    gw = [{"w": w["word"], "s": scene_start + w["start_ms"], "e": scene_start + w["end_ms"]} for w in words]
    groups = _group_words(gw, max_words, max_chars)

    entries = []
    for gi, g in enumerate(groups):
        start = g[0]["s"]
        next_start = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else scene_end
        # stay until the next chunk starts, but don't linger over a long pause
        end = min(next_start, g[-1]["e"] + hold_ms)
        end = max(end, min(start + 300, next_start))   # min readable time, never overlap the next chunk
        entries.append({
            "start_ms": start,
            "end_ms": end,
            "text": _wrap_text(" ".join(x["w"] for x in g), max_chars),
            "words": [{"w": x["w"], "s": x["s"], "e": x["e"]} for x in g],
        })
    return entries


def load_subtitles(srt_path: str) -> list[dict]:
    """Parse the SRT; if a matching .words.json exists, return those entries (with word timings)."""
    entries = parse_srt(srt_path)
    wj = Path(srt_path).with_suffix(".words.json")
    if wj.exists():
        try:
            with open(wj, "r", encoding="utf-8") as f:
                rich = json.load(f)
            if len(rich) == len(entries):
                return rich
        except Exception:
            pass
    return entries


def _split_into_chunks(text: str, total_ms: int, max_words: int) -> list[dict]:
    """Split narration into word-chunks with timing distributed proportionally."""
    words = text.split()
    if not words:
        return [{"text": "", "offset_ms": 0, "duration_ms": total_ms}]

    # Group words into chunks of max_words
    chunks_text = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks_text.append(" ".join(current))
            current = []
    if current:
        chunks_text.append(" ".join(current))

    total_words = len(words)
    result = []
    offset_ms = 0

    for i, chunk_text in enumerate(chunks_text):
        chunk_words = len(chunk_text.split())
        if i == len(chunks_text) - 1:
            chunk_ms = total_ms - offset_ms  # last chunk takes remaining time
        else:
            chunk_ms = int((chunk_words / total_words) * total_ms)

        result.append({
            "text": chunk_text,
            "offset_ms": offset_ms,
            "duration_ms": max(300, chunk_ms),  # min 300ms so it's readable
        })
        offset_ms += chunk_ms

    return result


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp: HH:MM:SS,mmm"""
    ms = max(0, ms)
    total_s, millis = divmod(ms, 1000)
    total_m, secs = divmod(total_s, 60)
    hours, mins = divmod(total_m, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Legacy: Whisper-based transcription (kept as fallback only)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, config: dict | None = None, safe_name: str | None = None) -> str:
    """[LEGACY] Transcribe audio via Whisper API. Use generate_srt_from_script() instead."""
    from openai import OpenAI

    if config is None:
        config = load_config()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    whisper_cfg = config["whisper"]

    audio_path = Path(audio_path)
    if safe_name is None:
        safe_name = audio_path.stem.replace("_full", "")

    srt_path = BASE_DIR / "subtitles" / f"{safe_name}.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Whisper] Transcribing: {audio_path.name}")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=whisper_cfg["model"],
            file=audio_file,
            language=whisper_cfg["language"],
            response_format="srt",
        )

    srt_content = transcript if isinstance(transcript, str) else str(transcript)
    srt_content = _wrap_srt_lines(srt_content, config["subtitle"]["max_chars_per_line"])

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"[Whisper] Subtitle saved -> {srt_path}")
    return str(srt_path)


def _wrap_srt_lines(srt_content: str, max_chars: int) -> str:
    """Wrap long lines in a Whisper-generated SRT string."""
    lines = srt_content.split("\n")
    result = []
    for line in lines:
        if re.match(r"^\d+$", line.strip()) or re.match(r"^\d{2}:", line) or not line.strip():
            result.append(line)
        elif len(line) > max_chars:
            result.append(_wrap_text(line, max_chars))
        else:
            result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _wrap_text(text: str, max_chars: int) -> str:
    """Wrap text at word boundaries, max 2 lines."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) >= 2:
            break
    if current and len(lines) < 2:
        lines.append(current)
    return "\n".join(lines[:2])


def parse_srt(srt_path: str) -> list[dict]:
    """Parse SRT file into list of {index, start_ms, end_ms, text}."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1].strip(),
        )
        if not time_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
        entries.append({
            "index": index,
            "start_ms": (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1,
            "end_ms":   (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2,
            "text": "\n".join(lines[2:]),
        })

    return entries


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    if len(sys.argv) > 1:
        entries = parse_srt(sys.argv[1])
        print(f"Parsed {len(entries)} subtitle entries from {sys.argv[1]}")
    else:
        print("Usage: python transcriber.py <path_to.srt>")