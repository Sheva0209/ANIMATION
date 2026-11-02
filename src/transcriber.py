"""
transcriber.py
Generate .srt subtitle from LLM script + TTS timing data (no Whisper needed).
Each scene narration is split into small word-chunks for natural scrolling subtitles.
"""

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
    Build a perfectly-synced SRT from LLM narration + TTS timing.
    Each scene is split into small word-chunks for natural subtitle scrolling.
    """
    if config is None:
        config = load_config()

    max_chars = config["subtitle"]["max_chars_per_line"]
    max_words = config["subtitle"].get("max_words_per_chunk", 7)

    scene_audio = audio_data["scene_audio"]
    scene_offsets = audio_data["scene_offsets"]

    if safe_name is None:
        safe_name = (
            re.sub(r"[^\w\s-]", "", script.get("title", "output"))
            .strip().replace(" ", "_").lower()[:60]
        )

    srt_path = BASE_DIR / "subtitles" / f"{safe_name}.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    srt_blocks = []
    entry_idx = 1

    for scene_info, start_ms in zip(scene_audio, scene_offsets):
        text = scene_info["narration"].strip()
        total_ms = scene_info["duration_ms"]

        chunks = _split_into_chunks(text, total_ms, max_words)
        for chunk in chunks:
            chunk_start = start_ms + chunk["offset_ms"]
            chunk_end = chunk_start + chunk["duration_ms"]
            chunk_text = _wrap_text(chunk["text"], max_chars)

            srt_blocks.append(
                f"{entry_idx}\n"
                f"{_ms_to_srt_time(chunk_start)} --> {_ms_to_srt_time(chunk_end)}\n"
                f"{chunk_text}\n"
            )
            entry_idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    print(f"[Subtitle] OK Synced SRT from script -> {srt_path} ({entry_idx - 1} entries)")
    return str(srt_path)


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
