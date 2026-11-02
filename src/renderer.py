"""
renderer.py
Pillow compositor + FFmpeg encoder
Renders scene-by-scene frames and produces final MP4 (landscape + portrait)
"""

import math
import os
import random
import re
import shutil
import subprocess
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).parent.parent
POSES_DIR = BASE_DIR / "poses"
BG_DIR = BASE_DIR / "backgrounds"
ASSETS_DIR = BASE_DIR / "assets"


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Font Loading ────────────────────────────────────────────────────────────

def load_font(config: dict, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = BASE_DIR / config["subtitle"]["font"]
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    # Fallback: try system fonts
    fallbacks = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fb in fallbacks:
        if Path(fb).exists():
            return ImageFont.truetype(fb, size)
    return ImageFont.load_default()


# ─── Image Utilities ─────────────────────────────────────────────────────────

def _coverfit(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize+crop an image so it fully covers a width×height canvas, centered."""
    img_ratio = img.width / img.height
    canvas_ratio = width / height
    if img_ratio > canvas_ratio:
        new_h = height
        new_w = int(height * img_ratio)
    else:
        new_w = width
        new_h = int(width / img_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return img.crop((left, top, left + width, top + height))


def load_background(bg_name: str, config: dict, width: int, height: int) -> Image.Image:
    bg_cfg = config["background_map"][bg_name]
    bg_path = BG_DIR / bg_cfg["file"]
    img = Image.open(bg_path).convert("RGB")
    return _coverfit(img, width, height)


def load_background_from_path(path: str, width: int, height: int) -> Image.Image:
    """Load a custom/uploaded background image (scene-specific, not from config background_map)."""
    img = Image.open(path).convert("RGB")
    return _coverfit(img, width, height)


def is_no_pose(pose_name) -> bool:
    """True when a scene deliberately wants no character -- background-only shot."""
    return not pose_name or str(pose_name).strip().lower() in ("none", "no pose", "no_pose")


def load_character(pose_name: str, config: dict, canvas_h: int) -> Image.Image:
    # Find the file with matching name (case-insensitive, ignore extension)
    target = pose_name.lower()
    char_img = None
    for p in POSES_DIR.iterdir():
        if p.stem.lower() == target and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            char_img = Image.open(p).convert("RGBA")
            break
    if char_img is None:
        raise FileNotFoundError(f"Pose not found: '{pose_name}' in {POSES_DIR}")

    scale = config["character"]["default_scale"]
    target_h = int(canvas_h * scale)
    aspect = char_img.width / char_img.height
    target_w = int(target_h * aspect)
    return char_img.resize((target_w, target_h), Image.LANCZOS)


def composite_frame(
    bg: Image.Image,
    char: Image.Image,
    position: str,
    canvas_w: int,
    canvas_h: int,
    char_alpha: float = 1.0,
    char_x_offset: int = 0,
) -> Image.Image:
    """Composite character onto background with positioning."""
    frame = bg.copy().convert("RGBA")

    # Character x position
    if position == "center":
        char_x = (canvas_w - char.width) // 2
    elif position == "left":
        char_x = int(canvas_w * 0.03)
    elif position == "right":
        char_x = canvas_w - char.width - int(canvas_w * 0.03)
    else:
        char_x = (canvas_w - char.width) // 2

    char_x += char_x_offset
    # Bottom-aligned character
    char_y = canvas_h - char.height

    # Apply alpha for fade in/out
    if char_alpha < 1.0:
        r, g, b, a = char.split()
        a = a.point(lambda p: int(p * char_alpha))
        char = Image.merge("RGBA", (r, g, b, a))

    frame.paste(char, (char_x, char_y), char)
    return frame.convert("RGB")


# ─── Subtitle Rendering ───────────────────────────────────────────────────────

def draw_subtitle(
    frame: Image.Image,
    text: str,
    config: dict,
    is_landscape: bool,
    font: ImageFont.FreeTypeFont,
    emotion: str = "neutral",
) -> Image.Image:
    if not text.strip():
        return frame

    sub_cfg = config["subtitle"]
    emotion_styles = sub_cfg.get("emotion_styles", {})

    # Get emotion style, fallback to neutral
    style = emotion_styles.get(emotion, emotion_styles.get("neutral", {}))
    scale = style.get("scale", 1.0)
    text_color = tuple(style.get("text_color", sub_cfg["text_color"]))
    bg_color_raw = style.get("bg_color", sub_cfg["bg_color"])

    draw = ImageDraw.Draw(frame)
    W, H = frame.size

    y_ratio = sub_cfg["landscape_y_ratio"] if is_landscape else sub_cfg["portrait_y_ratio"]
    pad_x = sub_cfg["padding_x"]
    pad_y = sub_cfg["padding_y"]
    outline_w = sub_cfg["outline_width"]
    outline_color = tuple(sub_cfg["outline_color"])
    radius = sub_cfg["border_radius"]

    # Scale font for emotion
    base_size = sub_cfg[f"font_size_{'landscape' if is_landscape else 'portrait'}"]
    actual_size = max(24, int(base_size * scale))
    if actual_size != base_size:
        scaled_font = load_font(config, actual_size)
    else:
        scaled_font = font

    lines = text.split("\n")

    # Calculate text dimensions
    line_bboxes = [draw.textbbox((0, 0), line, font=scaled_font) for line in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths = [bb[2] - bb[0] for bb in line_bboxes]
    line_spacing = 8

    total_text_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    max_text_w = max(line_widths) if line_widths else 0

    box_w = max_text_w + pad_x * 2
    box_h = total_text_h + pad_y * 2
    box_x = (W - box_w) // 2
    box_y = int(H * y_ratio) - box_h

    # Emotion: add left accent bar for non-neutral
    accent_w = 5 if emotion not in ("neutral", "happy") else 0

    # Draw rounded rect background
    bg_color = tuple(bg_color_raw)
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [box_x - accent_w, box_y, box_x + box_w + accent_w, box_y + box_h],
        radius=radius,
        fill=bg_color,
    )
    # Accent bar (colored left stripe for emphasis emotions)
    if accent_w > 0:
        accent_color = (*text_color[:3], 220)
        ov_draw.rounded_rectangle(
            [box_x - accent_w, box_y, box_x - 1, box_y + box_h],
            radius=4,
            fill=accent_color,
        )
    frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(frame)

    # Draw each line of text
    current_y = box_y + pad_y
    for i, line in enumerate(lines):
        lw = line_widths[i] if i < len(line_widths) else 0
        lx = (W - lw) // 2

        # Outline
        for ox in range(-outline_w, outline_w + 1):
            for oy in range(-outline_w, outline_w + 1):
                if ox == 0 and oy == 0:
                    continue
                draw.text((lx + ox, current_y + oy), line, font=scaled_font, fill=outline_color)

        # Main text
        draw.text((lx, current_y), line, font=scaled_font, fill=text_color)
        current_y += line_heights[i] + line_spacing

    return frame


# ─── Ken Burns Effect ─────────────────────────────────────────────────────────

def ken_burns_crop(bg: Image.Image, progress: float, canvas_w: int, canvas_h: int) -> Image.Image:
    """Slow zoom-in effect: progress 0.0 -> 1.0 maps 100% -> 110% zoom."""
    zoom = 1.0 + 0.10 * progress
    src_w = int(canvas_w / zoom)
    src_h = int(canvas_h / zoom)
    x_off = (bg.width - src_w) // 2
    y_off = (bg.height - src_h) // 2
    cropped = bg.crop((x_off, y_off, x_off + src_w, y_off + src_h))
    return cropped.resize((canvas_w, canvas_h), Image.LANCZOS)


# ─── Transition Frames ────────────────────────────────────────────────────────

def fade_transition(frame_a: Image.Image, frame_b: Image.Image, t: float) -> Image.Image:
    """Blend two frames: t=0 -> frame_a, t=1 -> frame_b."""
    return Image.blend(frame_a, frame_b, t)


# ─── Main Render Function ─────────────────────────────────────────────────────

def render_video(
    script: dict,
    audio_data: dict,
    srt_path: str,
    config: dict | None = None,
    safe_name: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Full render pipeline: generates landscape + portrait MP4.
    progress_callback(pct: int, detail: str) is called after each scene if provided.
    Returns dict with output paths.
    """
    if config is None:
        config = load_config()

    from transcriber import parse_srt

    if safe_name is None:
        safe_name = re.sub(r"[^\w\s-]", "", script["title"]).strip().replace(" ", "_").lower()[:60]

    fps = config["video"]["fps"]
    transition_frames = int(config["video"]["transition_duration"] * fps)
    entry_frames = int(config["character"]["entry_duration"] * fps)
    blink_min = config["character"]["blink_interval_min"]
    blink_max = config["character"]["blink_interval_max"]
    blink_hold_frames = 3  # ~0.1s blink

    subtitles = parse_srt(srt_path)

    outputs = {}

    portrait_only = config.get("video", {}).get("portrait_only", False)
    orientations = ["portrait"] if portrait_only else ["landscape", "portrait"]

    for orientation in orientations:

        W, H = config["video"][orientation]
        is_landscape = orientation == "landscape"
        font_size = config["subtitle"][f"font_size_{orientation}"]
        font = load_font(config, font_size)

        frames_dir = BASE_DIR / "output" / orientation / "frames" / safe_name
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Renderer] Rendering {orientation} ({W}×{H}) -- {len(script['scenes'])} scenes")

        global_frame_idx = 0

        # Pre-load all backgrounds and characters
        bgs = {}
        chars = {}
        blink_pairs = {}
        generated_images = {}   # scene_id -> PIL.Image for Gemini-generated scenes
        custom_bgs = {}         # scene_id -> PIL.Image for user-uploaded custom backgrounds

        for scene in script["scenes"]:
            bg_name = scene.get("background")
            pose_name = scene.get("pose")
            custom_bg_path = scene.get("custom_background_path")

            # Handle Gemini-generated scene images
            if scene.get("generated_image_path"):
                gen_path = scene["generated_image_path"]
                if Path(gen_path).exists():
                    gen_img = Image.open(gen_path).convert("RGB")
                    generated_images[scene["id"]] = _coverfit(gen_img, W, H)
                    continue  # no need to load bg/char for generated scene
                else:
                    print(f"[Renderer] WARNING: generated image not found: {gen_path}")

            # Handle user-uploaded custom background image (per-scene)
            if custom_bg_path:
                if Path(custom_bg_path).exists():
                    custom_bgs[scene["id"]] = load_background_from_path(custom_bg_path, W, H)
                else:
                    print(f"[Renderer] WARNING: custom background not found: {custom_bg_path}")
            elif bg_name and bg_name not in bgs:
                bgs[bg_name] = load_background(bg_name, config, W, H)

            # "No pose" scenes intentionally have no pose_name -- skip character loading
            if not is_no_pose(pose_name) and pose_name not in chars:
                chars[pose_name] = load_character(pose_name, config, H)

            # Pre-load blink pair if exists
            pose_cfg = config["pose_map"].get(pose_name or "", {})
            if pose_cfg.get("has_blink") and pose_cfg.get("blink_pair"):
                blink_name = pose_cfg["blink_pair"]
                if blink_name not in chars:
                    try:
                        chars[blink_name] = load_character(blink_name, config, H)
                        blink_pairs[pose_name] = blink_name
                    except FileNotFoundError:
                        print(f"[Renderer] Blink pair '{blink_name}' not found, skipping blink")

        # ── Pass 1: resolve bg/char/blink/timing for every scene (no pixels yet) ──
        scene_assets = []
        for scene_idx, scene in enumerate(script["scenes"]):
            pose_name = scene.get("pose")
            bg_name = scene.get("background")
            custom_bg_path = scene.get("custom_background_path")
            position = scene.get("character_position", "center")
            subtitle_emotion = scene.get("subtitle_emotion", "neutral")
            is_generated = scene["id"] in generated_images
            has_custom_bg = scene["id"] in custom_bgs
            no_pose = is_no_pose(pose_name)

            # Duration in frames (use actual audio timing if available)
            if scene_idx < len(audio_data["scene_audio"]):
                duration_ms = audio_data["scene_audio"][scene_idx]["duration_ms"]
            else:
                duration_ms = int(scene["estimated_duration"] * 1000)

            total_scene_frames = int((duration_ms / 1000) * fps)
            if total_scene_frames < 1:
                total_scene_frames = int(scene["estimated_duration"] * fps)

            pose_label = "NO POSE" if no_pose else pose_name
            bg_label = "CUSTOM-UPLOAD" if has_custom_bg else (bg_name or "AI-IMG")
            print(f"[Renderer] Scene {scene['id']}: '{pose_label}' on '{bg_label}' -- {total_scene_frames} frames")

            # Setup bg, char, blink for this scene
            gen_bg = None
            if is_generated:
                gen_bg = generated_images[scene["id"]]
                bg = None
                char = None
                blink_char = None
            else:
                if has_custom_bg:
                    bg = custom_bgs[scene["id"]]
                else:
                    bg = bgs.get(bg_name)
                    if bg is None:
                        print(f"[Renderer] WARNING: missing bg '{bg_name}', using first available")
                        bg = next(iter(bgs.values()))

                if no_pose:
                    # Deliberately no character -- background-only shot
                    char = None
                    blink_char = None
                else:
                    char = chars.get(pose_name)
                    if char is None:
                        print(f"[Renderer] WARNING: missing pose '{pose_name}', using first available")
                        char = next(iter(chars.values()))
                    blink_char = chars.get(blink_pairs.get(pose_name))

            # Schedule blink events for this scene
            blink_events = set()
            if blink_char:
                t = random.uniform(blink_min, blink_max) * fps
                while t < total_scene_frames - blink_hold_frames:
                    blink_events.update(range(int(t), int(t) + blink_hold_frames))
                    t += random.uniform(blink_min, blink_max) * fps

            scene_offset_ms = (
                audio_data["scene_offsets"][scene_idx]
                if scene_idx < len(audio_data["scene_offsets"]) else 0
            )

            scene_assets.append({
                "is_generated": is_generated,
                "gen_bg": gen_bg,
                "bg": bg,
                "char": char,
                "blink_char": blink_char,
                "blink_events": blink_events,
                "position": position,
                "subtitle_emotion": subtitle_emotion,
                "total_scene_frames": total_scene_frames,
                "scene_offset_ms": scene_offset_ms,
            })

        def render_base_frame(assets: dict, local_f: int) -> Image.Image:
            """Background (Ken Burns) + character (entry fade / blink) -- no subtitle, no transition."""
            total = assets["total_scene_frames"]
            progress = local_f / max(total - 1, 1)
            source_bg = assets["gen_bg"] if assets["is_generated"] else assets["bg"]
            kb_bg = ken_burns_crop(source_bg, progress, W, H)

            char = assets["char"]
            if char is not None:
                char_alpha = (local_f / entry_frames) if local_f < entry_frames else 1.0
                active_char = (
                    assets["blink_char"]
                    if (assets["blink_char"] and local_f in assets["blink_events"])
                    else char
                )
                return composite_frame(kb_bg, active_char, assets["position"], W, H, char_alpha, 0)
            return kb_bg.copy()

        def save_frame(img: Image.Image, idx: int) -> None:
            img.save(str(frames_dir / f"frame_{idx:06d}.jpg"), "JPEG", quality=92)

        # Crossfade length per boundary, capped so it never outruns either neighboring scene
        n_scenes = len(scene_assets)
        boundary_trans = [
            max(0, min(transition_frames, scene_assets[i]["total_scene_frames"], scene_assets[i + 1]["total_scene_frames"]))
            for i in range(n_scenes - 1)
        ]

        # ── Pass 2: write frames -- every scene keeps its FULL frame count (stays in sync
        # with the audio); only the pixel content of each scene's tail is replaced by a
        # crossfade into a preview of the next scene, instead of fading to black. ──
        for scene_idx, assets in enumerate(scene_assets):
            total = assets["total_scene_frames"]
            has_next = scene_idx < n_scenes - 1
            tfrm = boundary_trans[scene_idx] if has_next else 0
            next_assets = scene_assets[scene_idx + 1] if has_next else None
            # The dissolve target is next scene's own opening frame (local_f=0), held static
            # for the whole crossfade -- NOT an animating preview. If it animated here too,
            # scene B's own loop would then restart its Ken Burns zoom / character fade-in
            # from scratch right after, which plays back as a visible "rewind" at the cut.
            # Dissolving into a static frame 0 means the handoff into scene B's real frame 0
            # is seamless (it's the same image the blend was already converging toward).
            next_first_frame = render_base_frame(next_assets, 0) if has_next and tfrm > 0 else None

            for local_f in range(total):
                frame = render_base_frame(assets, local_f)

                # Crossfade into the next scene during this scene's last `tfrm` frames --
                # no black flash, and total frame count (= audio duration) is untouched.
                if has_next and tfrm > 0 and local_f >= total - tfrm:
                    t = local_f - (total - tfrm)       # 0 .. tfrm-1
                    blend_amt = (t + 1) / tfrm           # ramps smoothly toward the next scene
                    frame = fade_transition(frame, next_first_frame, blend_amt)

                current_time_ms = assets["scene_offset_ms"] + int((local_f / fps) * 1000)
                subtitle_text = get_subtitle_at(subtitles, current_time_ms)
                if subtitle_text:
                    frame = draw_subtitle(frame, subtitle_text, config, is_landscape, font, assets["subtitle_emotion"])
                save_frame(frame, global_frame_idx)
                global_frame_idx += 1

                if local_f % (fps * 5) == 0:
                    print(f"[Renderer]   {local_f}/{total} frames done")

            if progress_callback:
                pct = 33 + int((scene_idx + 1) / n_scenes * 57)
                progress_callback(pct, f"Scene {scene_idx + 1}/{n_scenes} -- {global_frame_idx} frames")

        print(f"[Renderer] Total frames rendered: {global_frame_idx}")

        # FFmpeg: frames + audio -> MP4
        out_path = BASE_DIR / "output" / orientation / f"{safe_name}.mp4"
        _ffmpeg_encode(frames_dir, audio_data["full_audio"], out_path, fps, srt_path, config)
        outputs[orientation] = str(out_path)
        print(f"[Renderer] OK {orientation} video -> {out_path}")

        # Cleanup frames
        shutil.rmtree(frames_dir)

    return outputs


def get_subtitle_at(subtitles: list[dict], time_ms: int) -> str:
    for sub in subtitles:
        if sub["start_ms"] <= time_ms <= sub["end_ms"]:
            return sub["text"]
    return ""


def _ffmpeg_encode(
    frames_dir: Path,
    audio_path: str,
    out_path: Path,
    fps: int,
    srt_path: str,
    config: dict,
) -> None:
    """Use FFmpeg to encode frames + audio into MP4."""
    frames_pattern = str(frames_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        # Normalize to 44100Hz stereo -- prevents noise when concat with asset clips
        "-ar", "44100",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]


    print(f"[FFmpeg] Encoding: {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FFmpeg] STDERR:\n{result.stderr}")
        raise RuntimeError(f"FFmpeg encoding failed: {result.returncode}")
    print(f"[FFmpeg] Encoding complete: {out_path}")


if __name__ == "__main__":
    import json
    import sys
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")

    if len(sys.argv) < 4:
        print("Usage: python renderer.py <script.json> <audio_data.json> <subtitle.srt>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        script = json.load(f)
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    cfg = load_config()
    outputs = render_video(script, audio_data, sys.argv[3], cfg)
    print("Output files:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")