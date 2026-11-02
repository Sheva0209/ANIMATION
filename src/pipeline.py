"""
pipeline.py
Orchestrates the full animation pipeline:
  1. Script generation (LLM)
  2. TTS voice-over (ElevenLabs)
  3. Subtitle generation (from script -- no Whisper needed)
  4. Video rendering (Pillow + FFmpeg)
"""

import json
import os
import re
import sys
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

# Ensure src modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from script_generator import generate_script, save_script
from tts_generator import generate_tts
from transcriber import generate_srt_from_script
from scene_generator import process_generated_scenes
from renderer import render_video


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def topic_to_safe_name(topic: str) -> str:
    return re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()[:60]


def run_pipeline(
    topic: str,
    skip_tts: bool = False,
    script_path: str | None = None,
    audio_path: str | None = None,
    srt_path: str | None = None,
) -> dict:
    """
    Run the full animation pipeline for a given topic.

    Args:
        topic:       The video topic/title
        skip_tts:    Skip TTS generation (use existing audio_path)
        script_path: Override: use existing script JSON instead of generating
        audio_path:  Override: use existing full audio MP3
        srt_path:    Override: use existing SRT file (skips script-based generation)

    Returns:
        dict with output video paths and intermediate file paths
    """
    load_dotenv(BASE_DIR / ".env")
    config = load_config()
    safe_name = topic_to_safe_name(topic)

    print(f"\n{'='*60}")
    print(f"  SHEVA ANIMATION PIPELINE")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # --- Step 1: Script Generation ---
    print("--- STEP 1/4: Script Generation ---")
    if script_path and Path(script_path).exists():
        print(f"[Pipeline] Using existing script: {script_path}")
        with open(script_path, "r", encoding="utf-8") as f:
            script = json.load(f)
    else:
        script = generate_script(topic, config)
        script_path = str(save_script(script, topic))

    total_scenes = len(script["scenes"])
    estimated_duration = sum(s["estimated_duration"] for s in script["scenes"])
    print(f"[Pipeline] OK - Script ready: {total_scenes} scenes, ~{estimated_duration:.0f}s")

    # --- Step 2: TTS Generation ---
    print("\n--- STEP 2/4: TTS Voice-Over ---")
    checkpoint_path = BASE_DIR / "audio" / f"{safe_name}_checkpoint.json"

    if skip_tts and audio_path and Path(audio_path).exists():
        print(f"[Pipeline] Using existing audio: {audio_path}")
        from pydub import AudioSegment
        full_seg = AudioSegment.from_mp3(audio_path)
        offsets, current, scene_audio = [], 0, []
        for scene in script["scenes"]:
            offsets.append(current)
            est_ms = int(scene["estimated_duration"] * 1000)
            scene_audio.append({
                "scene_id": scene["id"],
                "path": audio_path,
                "duration_ms": est_ms,
                "narration": scene["narration"],
            })
            current += est_ms
        audio_data = {
            "full_audio": audio_path,
            "scene_audio": scene_audio,
            "scene_offsets": offsets,
            "total_duration_ms": len(full_seg),
        }
    elif checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if all(Path(sa["path"]).exists() for sa in cached.get("scene_audio", [])):
                print(f"[Pipeline] [RESUME] TTS checkpoint found, skipping TTS generation")
                audio_data = cached
                audio_path = audio_data["full_audio"]
            else:
                raise ValueError("audio files missing from checkpoint")
        except Exception as e:
            print(f"[Pipeline] Checkpoint invalid ({e}), re-generating TTS")
            audio_data = generate_tts(script, config, safe_name)
            audio_path = audio_data["full_audio"]
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(audio_data, f, ensure_ascii=False)
    else:
        audio_data = generate_tts(script, config, safe_name)
        audio_path = audio_data["full_audio"]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(audio_data, f, ensure_ascii=False)
        print(f"[Pipeline] [OK] TTS checkpoint saved -> {checkpoint_path.name}")

    total_duration = audio_data["total_duration_ms"] / 1000
    print(f"[Pipeline] OK - Audio ready: {total_duration:.1f}s total")

    # --- Step 3: Subtitle Generation (from script -- no Whisper) ---
    print("\n--- STEP 3/4: Subtitle Generation (from script) ---")
    if srt_path and Path(srt_path).exists():
        print(f"[Pipeline] Using existing SRT: {srt_path}")
    else:
        srt_path = generate_srt_from_script(script, audio_data, config, safe_name)

    print(f"[Pipeline] OK - Subtitles ready: {srt_path}")

    # --- Step 3.5: AI Scene Generation (Gemini Imagen) ---
    has_generated_scenes = any(s.get("generated_scene") for s in script["scenes"])
    if has_generated_scenes:
        print("\n--- STEP 3.5/4: AI Scene Generation (Gemini Imagen) ---")
        script = process_generated_scenes(script, safe_name, config)
        gen_count = sum(1 for s in script["scenes"] if s.get("generated_image_path"))
        print(f"[Pipeline] OK - Generated {gen_count} AI scene images")
    else:
        print("[Pipeline] No AI-generated scenes in this script, skipping Gemini step")

    # --- Step 4: Video Rendering ---
    print("\n--- STEP 4/4: Video Rendering ---")
    outputs = render_video(script, audio_data, srt_path, config, safe_name)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Landscape: {outputs.get('landscape', 'N/A')}")
    print(f"  Portrait:  {outputs.get('portrait', 'N/A')}")
    print(f"  Script:    {script_path}")
    print(f"  Audio:     {audio_path}")
    print(f"  Subtitles: {srt_path}")
    print(f"{'='*60}\n")

    return {
        "landscape": outputs.get("landscape"),
        "portrait": outputs.get("portrait"),
        "script": script_path,
        "audio": audio_path,
        "srt": srt_path,
        "duration_seconds": total_duration,
    }


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "tips nggak burnout saat UTS"
    result = run_pipeline(topic)
