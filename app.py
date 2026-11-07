"""
app.py — Flask web server for Sheva Animation Studio
Provides REST API + SSE for the web UI
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env", override=True)

import yaml
from script_generator import generate_script, save_script, load_config

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB upload limit

# Global state per job
jobs: dict[str, dict] = {}
progress_queues: dict[str, queue.Queue] = {}

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── Helpers ────────────────────────────────────────────────────────────────

def topic_to_safe_name(topic: str) -> str:
    return re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()[:60]


def get_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def emit(job_id: str, event: str, data: dict):
    """Push SSE event to a job's queue."""
    if job_id in progress_queues:
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        progress_queues[job_id].put(msg)


# ── API Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(BASE_DIR / "static" / "index.html")


@app.route("/api/config")
def api_config():
    """Return poses, backgrounds, emotions, and app config for UI dropdowns."""
    cfg = get_config()
    poses = {
        name: data for name, data in cfg["pose_map"].items()
        if not data.get("is_blink_frame")
    }
    bgs = cfg["background_map"]
    emotions = list(cfg["subtitle"].get("emotion_styles", {}).keys())
    return jsonify({
        "poses": poses,
        "backgrounds": bgs,
        "emotions": emotions,
        "language": cfg["llm"].get("language", "en"),
        "niche": cfg["llm"].get("niche", "song_meaning"),
        "portrait_only": cfg.get("video", {}).get("portrait_only", False),
    })


@app.route("/api/generate-script", methods=["POST"])
def api_generate_script():
    """Step 1: Generate script from topic via GPT-4o.
    Accepts optional niche and language overrides from UI.
    """
    body = request.get_json()
    topic    = (body or {}).get("topic", "").strip()
    niche    = (body or {}).get("niche", "").strip()
    language = (body or {}).get("language", "").strip()

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    cfg = get_config()

    # Apply UI overrides for niche and language (per-request, does not modify config.yaml)
    if niche and niche in ("science_facts", "song_meaning", "education"):
        cfg["llm"]["niche"] = niche
    if language and language in ("en", "id"):
        cfg["llm"]["language"] = language

    try:
        script = generate_script(topic, cfg)
        script_path = str(save_script(script, topic))
        return jsonify({
            "script": script,
            "script_path": script_path,
            "safe_name": topic_to_safe_name(topic),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-asset", methods=["POST"])
def api_upload_asset():
    """Upload an image or video asset to be inserted between scenes."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    upload_dir = BASE_DIR / "assets" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Safe unique filename
    clean_name = re.sub(r"[^\w.-]", "_", f.filename)
    safe_filename = f"{int(time.time() * 1000)}_{clean_name}"
    save_path = upload_dir / safe_filename
    f.save(str(save_path))

    file_type = "image" if ext in ALLOWED_IMAGE_EXT else "video"
    return jsonify({
        "path": str(save_path),
        "filename": f.filename,
        "safe_filename": safe_filename,
        "type": file_type,
        "url": f"/api/asset-preview/{safe_filename}",
    })


@app.route("/api/asset-preview/<filename>")
def api_asset_preview(filename: str):
    """Serve uploaded asset for preview."""
    return send_from_directory(BASE_DIR / "assets" / "uploads", filename)


@app.route("/api/render", methods=["POST"])
def api_render():
    """
    Step 2: Start full render pipeline with (possibly edited) script.
    insert_assets: list of {after_scene_idx, path, duration, type}
    Returns job_id immediately; client polls /api/progress/<job_id> for SSE.
    """
    body = request.get_json()
    script = body.get("script")
    topic = body.get("topic", "").strip()
    insert_assets = body.get("insert_assets", [])  # NEW: manual asset insertions

    if not script or not topic:
        return jsonify({"error": "script and topic required"}), 400

    job_id = f"{int(time.time() * 1000)}"
    safe_name = topic_to_safe_name(topic)

    jobs[job_id] = {
        "status": "queued",
        "topic": topic,
        "safe_name": safe_name,
        "outputs": {},
        "error": None,
    }
    progress_queues[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, script, topic, safe_name, insert_assets),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    """SSE stream of pipeline progress for a given job."""
    def generate():
        q = progress_queues.get(job_id)
        if q is None:
            yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
                yield msg
                if '"done"' in msg or '"error"' in msg:
                    break
            except queue.Empty:
                yield ": heartbeat\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/outputs/<path:filename>")
def api_output_file(filename: str):
    """Serve output video files."""
    return send_from_directory(BASE_DIR / "output", filename)


@app.route("/api/outputs-list")
def api_outputs_list():
    """List all rendered videos."""
    out_dir = BASE_DIR / "output"
    results = []
    for orientation in ["landscape", "portrait"]:
        d = out_dir / orientation
        if d.exists():
            for f in sorted(d.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
                results.append({
                    "name": f.stem.replace("_", " ").title(),
                    "safe_name": f.stem,
                    "orientation": orientation,
                    "url": f"/api/outputs/{orientation}/{f.name}",
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "mtime": f.stat().st_mtime,
                })
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(results)


# ── Pipeline Thread ────────────────────────────────────────────────────────

def _run_pipeline_thread(job_id: str, script: dict, topic: str, safe_name: str, insert_assets: list):
    """Run TTS → Subtitle → Render in a background thread, emitting SSE progress."""
    from tts_generator import generate_tts
    from transcriber import generate_srt_from_script
    from renderer import render_video
    from scene_generator import process_generated_scenes

    cfg = get_config()
    jobs[job_id]["status"] = "running"

    try:
        # ── Step 1: TTS ───────────────────────────────────────────────────────
        emit(job_id, "step", {"step": 1, "total": 3, "label": "Generating voice-over (Chatterbox TTS)..."})
        emit(job_id, "progress", {"pct": 3, "detail": "Starting TTS..."})

        # Per-scene cache is handled inside generate_tts (hash of text + emotion + params + voice):
        # unchanged scenes are reused, edited scenes are regenerated.
        audio_data = generate_tts(script, cfg, safe_name)

        total_dur = audio_data["total_duration_ms"] / 1000
        emit(job_id, "step_done", {"step": 1, "label": f"Voice-over ready ({total_dur:.1f}s)"})
        emit(job_id, "progress", {"pct": 28, "detail": f"All {len(script['scenes'])} scenes generated"})

        # ── Step 2: Subtitles from script ───────────────────────────────────────
        emit(job_id, "step", {"step": 2, "total": 3, "label": "Generating subtitles (from script)..."})
        srt_path = generate_srt_from_script(script, audio_data, cfg, safe_name)
        emit(job_id, "step_done", {"step": 2, "label": "Subtitles ready"})
        emit(job_id, "progress", {"pct": 33, "detail": "Subtitles complete"})

        # ── Step 2.5: AI Scenes (optional) ──────────────────────────────────
        if any(s.get("generated_scene") for s in script["scenes"]):
            emit(job_id, "progress", {"pct": 35, "detail": "Generating AI scene images..."})
            script = process_generated_scenes(script, safe_name, cfg)

        # ── Step 3: Render ───────────────────────────────────────────────────
        emit(job_id, "step", {"step": 3, "total": 3, "label": "Rendering portrait video..."})

        def on_progress(pct, detail):
            emit(job_id, "progress", {"pct": pct, "detail": detail})

        outputs = render_video(script, audio_data, srt_path, cfg, safe_name, progress_callback=on_progress)
        portrait_path = outputs.get("portrait")

        # ── Step 3.5: Splice manual assets ─────────────────────────────────
        if insert_assets and portrait_path:
            emit(job_id, "progress", {"pct": 95, "detail": f"Splicing {len(insert_assets)} asset clip(s)..."})
            portrait_path = _splice_assets(portrait_path, insert_assets, audio_data, cfg, safe_name)
            emit(job_id, "progress", {"pct": 98, "detail": "Asset splice complete"})

        portrait_url = f"/api/outputs/portrait/{safe_name}.mp4"
        jobs[job_id]["status"] = "done"
        jobs[job_id]["outputs"] = {"portrait": portrait_url}
        emit(job_id, "step_done", {"step": 3, "label": "Portrait video encoded"})
        emit(job_id, "done", {"portrait": portrait_url})

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        emit(job_id, "error", {"message": str(e)})
        import traceback
        traceback.print_exc()


# ── Asset Splicing ─────────────────────────────────────────────────────────

def _splice_assets(video_path: str, insert_assets: list, audio_data: dict, cfg: dict, safe_name: str) -> str:
    """
    Splice manual asset clips (images/videos) into the rendered video at scene boundaries.
    Uses FFmpeg concat demuxer.
    """
    if not insert_assets:
        return video_path

    fps = cfg["video"]["fps"]
    W, H = cfg["video"]["portrait"]

    # Calculate cumulative scene end timestamps (seconds)
    scene_end_times = []
    t = 0.0
    for sa in audio_data["scene_audio"]:
        t += sa["duration_ms"] / 1000.0
        scene_end_times.append(t)

    temp_dir = BASE_DIR / "output" / "temp" / safe_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Sort assets by scene index
    sorted_assets = sorted(insert_assets, key=lambda x: x.get("after_scene_idx", 0))

    # Build list of video segments + asset clips in order
    segments = []  # each: {"type": "video"|"asset", ...}
    prev_time = 0.0

    for asset in sorted_assets:
        idx = asset.get("after_scene_idx", 0)
        if idx >= len(scene_end_times):
            idx = len(scene_end_times) - 1
        split_time = scene_end_times[idx]

        if split_time > prev_time:
            segments.append({"type": "video", "start": prev_time, "end": split_time})
        segments.append({
            "type": "asset",
            "path": asset["path"],
            "duration": float(asset.get("duration", 3)),
            "file_type": asset.get("file_type", "image"),
        })
        prev_time = split_time

    # Add remaining video segment
    segments.append({"type": "video", "start": prev_time, "end": None})

    # Create temp clips
    temp_clips = []
    for i, seg in enumerate(segments):
        clip_path = temp_dir / f"seg_{i:03d}.mp4"
        if seg["type"] == "video":
            _ffmpeg_trim(video_path, seg["start"], seg["end"], str(clip_path), W, H, fps)
        else:
            _ffmpeg_asset_clip(seg["path"], seg["duration"], seg["file_type"], W, H, fps, str(clip_path))
        temp_clips.append(clip_path)

    # Write FFmpeg concat list
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in temp_clips:
            f.write(f"file '{str(clip).replace(chr(92), '/')}'\n")

    # Final concat
    final_path = str(Path(video_path).parent / f"{safe_name}_final.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        final_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[AssetSplice] FFmpeg error:\n{result.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return video_path  # fallback to original

    # Replace original with spliced version
    shutil.move(final_path, video_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return video_path


def _ffmpeg_trim(src: str, start: float, end: float | None, out: str, W: int, H: int, fps: int):
    """
    Extract a segment from the main video (which has voice-over audio).
    Puts -ss BEFORE -i for fast keyframe-accurate seeking (avoids noise/artifacts).
    Normalizes audio to 44100Hz stereo AAC for consistent concat.
    """
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", src]
    if end is not None:
        duration = max(0.1, end - start)
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", f"scale={W}:{H},setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        # Normalize audio: 44100Hz, stereo, AAC — prevents noise from sample-rate mismatch
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)



def _ffmpeg_probe_has_audio(path: str) -> bool:
    """Return True if the media file has at least one audio stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return "audio" in result.stdout


def _ffmpeg_asset_clip(asset_path: str, duration: float, file_type: str, W: int, H: int, fps: int, out: str):
    """
    Convert an image or video asset to a portrait clip.
    - Images: static image + silence
    - Videos: original audio preserved (keep original sound!); silence added if no audio track
    All output clips normalized to 44100Hz stereo AAC to prevent noise during concat.
    """
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    AUDIO_NORM = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]

    if file_type == "image":
        # Static image + silence
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", asset_path,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-vf", vf,
            "-t", str(duration), "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            *AUDIO_NORM, "-shortest", out,
        ]
    else:
        # Video asset — check if it has audio
        has_audio = _ffmpeg_probe_has_audio(asset_path)

        if has_audio:
            # Keep original audio, re-encode to portrait + normalize
            cmd = [
                "ffmpeg", "-y", "-i", asset_path,
                "-vf", vf,
                "-t", str(duration), "-r", str(fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                *AUDIO_NORM, out,
            ]
        else:
            # No audio in video — add silence
            cmd = [
                "ffmpeg", "-y", "-i", asset_path,
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-vf", vf,
                "-t", str(duration), "-r", str(fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                *AUDIO_NORM, "-map", "0:v:0", "-map", "1:a:0", "-shortest", out,
            ]

    subprocess.run(cmd, capture_output=True, check=True)



# ── Dev Server ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    (BASE_DIR / "static").mkdir(exist_ok=True)
    (BASE_DIR / "assets" / "uploads").mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("  Sheva Animation Studio")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)