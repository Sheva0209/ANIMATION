"""
renderer.py
Pillow compositor + FFmpeg encoder
Renders scene-by-scene frames and produces final MP4 (landscape + portrait)
"""

import functools
import math
import os
import random
import re
import subprocess
import yaml
from concurrent.futures import ThreadPoolExecutor
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

@functools.lru_cache(maxsize=32)
def _load_font_cached(path_str: str, size: int):
    return ImageFont.truetype(path_str, size)


def load_font(config: dict, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = BASE_DIR / config["subtitle"]["font"]
    if font_path.exists():
        return _load_font_cached(str(font_path), size)
    # Fallback: try system fonts
    fallbacks = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fb in fallbacks:
        if Path(fb).exists():
            return _load_font_cached(fb, size)
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
    inplace: bool = False,
) -> Image.Image:
    """Composite character onto background with positioning.
    inplace=True pastes straight onto `bg` (only safe when bg is a throw-away frame)."""
    frame = bg if inplace else bg.copy()
    if frame.mode != "RGB":
        frame = frame.convert("RGB")

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
    return frame


# ─── Subtitle Rendering ───────────────────────────────────────────────────────

_SUB_CACHE: dict = {}


def _build_subtitle_overlay(text, config, is_landscape, emotion, W, H, font, active_idx=-1):
    """Render one subtitle chunk to a small RGBA overlay. Returns (overlay, (x, y))."""
    sub_cfg = config["subtitle"]
    emotion_styles = sub_cfg.get("emotion_styles", {})
    style = emotion_styles.get(emotion, emotion_styles.get("neutral", {}))
    scale = style.get("scale", 1.0)
    text_color = tuple(style.get("text_color", sub_cfg["text_color"]))
    bg_color = tuple(style.get("bg_color", sub_cfg["bg_color"]))

    y_ratio = sub_cfg["landscape_y_ratio"] if is_landscape else sub_cfg["portrait_y_ratio"]
    pad_x = sub_cfg["padding_x"]
    pad_y = sub_cfg["padding_y"]
    outline_w = sub_cfg["outline_width"]
    outline_color = tuple(sub_cfg["outline_color"])
    radius = sub_cfg["border_radius"]

    base_size = sub_cfg[f"font_size_{'landscape' if is_landscape else 'portrait'}"]
    actual_size = max(24, int(base_size * scale))
    scaled_font = load_font(config, actual_size) if actual_size != base_size else font

    lines = text.split("\n")
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    line_bboxes = [measure.textbbox((0, 0), line, font=scaled_font) for line in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths = [bb[2] - bb[0] for bb in line_bboxes]
    line_spacing = 8

    total_text_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    max_text_w = max(line_widths) if line_widths else 0

    box_w = max_text_w + pad_x * 2
    box_h = total_text_h + pad_y * 2
    box_x = (W - box_w) // 2
    box_y = int(H * y_ratio) - box_h

    # Left accent bar for emphasis emotions
    accent_w = 5 if emotion not in ("neutral", "happy") else 0

    ov_x = box_x - accent_w
    ov_w = box_w + 2 * accent_w + 1
    ov_h = box_h + 1

    overlay = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([0, 0, ov_w - 1, ov_h - 1], radius=radius, fill=bg_color)
    if accent_w > 0:
        od.rounded_rectangle([0, 0, accent_w - 1, ov_h - 1], radius=4, fill=(*text_color[:3], 220))

    text_layer = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    # Karaoke: the word being spoken gets the active colour. If that colour is too close to
    # the emotion's own text colour (e.g. yellow on "happy"), the active word goes white instead.
    active_color = text_color
    if active_idx >= 0:
        kcfg = sub_cfg.get("karaoke", {}) or {}
        active_color = tuple(kcfg.get("active_color", [255, 225, 70, 255]))
        if sum(abs(a - b) for a, b in zip(active_color[:3], text_color[:3])) < 130:
            active_color = (255, 255, 255, 255)

    current_y = pad_y
    word_no = 0
    for i, line in enumerate(lines):
        lx = (W - line_widths[i]) // 2 - ov_x
        if active_idx < 0:
            td.text((lx, current_y), line, font=scaled_font, fill=text_color,
                    stroke_width=outline_w, stroke_fill=outline_color)
        else:
            line_words = line.split(" ")
            for k, wd in enumerate(line_words):
                prefix = " ".join(line_words[:k]) + (" " if k else "")
                wx = lx + scaled_font.getlength(prefix)
                td.text((wx, current_y), wd, font=scaled_font,
                        fill=active_color if word_no == active_idx else text_color,
                        stroke_width=outline_w, stroke_fill=outline_color)
                word_no += 1
        current_y += line_heights[i] + line_spacing

    overlay = Image.alpha_composite(overlay, text_layer)
    return overlay, (ov_x, box_y)


def draw_subtitle(
    frame: Image.Image,
    text: str,
    config: dict,
    is_landscape: bool,
    font: ImageFont.FreeTypeFont,
    emotion: str = "neutral",
    active_idx: int = -1,
) -> Image.Image:
    """Paste a (cached) subtitle overlay onto `frame` in place and return it."""
    if not text.strip():
        return frame
    key = (text, emotion, is_landscape, frame.size, active_idx)
    entry = _SUB_CACHE.get(key)
    if entry is None:
        W, H = frame.size
        entry = _build_subtitle_overlay(text, config, is_landscape, emotion, W, H, font, active_idx)
        _SUB_CACHE[key] = entry
    overlay, pos = entry
    if frame.mode != "RGB":
        frame = frame.convert("RGB")
    frame.paste(overlay, pos, overlay)
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
    return cropped.resize((canvas_w, canvas_h), Image.BILINEAR)


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

    from transcriber import load_subtitles

    if safe_name is None:
        safe_name = re.sub(r"[^\w\s-]", "", script["title"]).strip().replace(" ", "_").lower()[:60]

    fps = config["video"]["fps"]
    transition_frames = int(config["video"]["transition_duration"] * fps)
    entry_frames = int(config["character"]["entry_duration"] * fps)
    blink_min = config["character"]["blink_interval_min"]
    blink_max = config["character"]["blink_interval_max"]
    blink_hold_frames = 3  # ~0.1s blink

    subtitles = load_subtitles(srt_path)
    karaoke = bool((config["subtitle"].get("karaoke", {}) or {}).get("enabled", True))

    outputs = {}

    portrait_only = config.get("video", {}).get("portrait_only", False)
    orientations = ["portrait"] if portrait_only else ["landscape", "portrait"]

    for orientation in orientations:

        W, H = config["video"][orientation]
        is_landscape = orientation == "landscape"
        font_size = config["subtitle"][f"font_size_{orientation}"]
        font = load_font(config, font_size)

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
                return composite_frame(kb_bg, active_char, assets["position"], W, H, char_alpha, 0, inplace=True)
            return kb_bg

        # Crossfade length per boundary, capped so it never outruns either neighboring scene
        n_scenes = len(scene_assets)
        boundary_trans = [
            max(0, min(transition_frames, scene_assets[i]["total_scene_frames"], scene_assets[i + 1]["total_scene_frames"]))
            for i in range(n_scenes - 1)
        ]

        def make_frame(assets, tfrm, has_next, next_first_frame, total, local_f):
            """One finished frame: base + crossfade (tail of scene) + subtitle. Thread-safe."""
            frame = render_base_frame(assets, local_f)
            # Crossfade into the next scene during this scene's last `tfrm` frames --
            # no black flash, and total frame count (= audio duration) is untouched.
            if has_next and tfrm > 0 and local_f >= total - tfrm:
                t = local_f - (total - tfrm)           # 0 .. tfrm-1
                frame = fade_transition(frame, next_first_frame, (t + 1) / tfrm)
            current_time_ms = assets["scene_offset_ms"] + int((local_f / fps) * 1000)
            entry = get_subtitle_entry_at(subtitles, current_time_ms)
            if entry and entry["text"]:
                active = _active_word_index(entry, current_time_ms) if karaoke else -1
                frame = draw_subtitle(frame, entry["text"], config, is_landscape, font,
                                      assets["subtitle_emotion"], active)
            return frame

        workers = int(config["video"].get("render_threads", 0) or 0) or min(8, os.cpu_count() or 1)
        out_path = BASE_DIR / "output" / orientation / f"{safe_name}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _SUB_CACHE.clear()
        print(f"[Renderer] Encoder: {config['video'].get('encoder', 'libx264')} | render threads: {workers}")
        pipe = _FFmpegPipe(W, H, fps, audio_data["full_audio"], out_path, config)

        # ── Pass 2: render frames (in order) and stream them straight into FFmpeg --
        # no JPEG files on disk. Every scene keeps its FULL frame count (stays in sync with
        # the audio); only the pixel content of each scene's tail is replaced by a crossfade.
        # The dissolve target is the next scene's own opening frame (local_f=0), held static,
        # so the handoff into scene B's real frame 0 is seamless (no "rewind" at the cut).
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                batch = max(8, workers * 4)   # bounded window keeps memory flat
                for scene_idx, assets in enumerate(scene_assets):
                    total = assets["total_scene_frames"]
                    has_next = scene_idx < n_scenes - 1
                    tfrm = boundary_trans[scene_idx] if has_next else 0
                    next_first_frame = (
                        render_base_frame(scene_assets[scene_idx + 1], 0) if has_next and tfrm > 0 else None
                    )
                    render = functools.partial(make_frame, assets, tfrm, has_next, next_first_frame, total)

                    for start in range(0, total, batch):
                        idxs = range(start, min(start + batch, total))
                        frames = pool.map(render, idxs) if workers > 1 else map(render, idxs)
                        for img in frames:
                            pipe.write(img)
                            global_frame_idx += 1

                    print(f"[Renderer]   scene {scene_idx + 1}/{n_scenes}: {global_frame_idx} frames so far")
                    if progress_callback:
                        pct = 33 + int((scene_idx + 1) / n_scenes * 57)
                        progress_callback(pct, f"Scene {scene_idx + 1}/{n_scenes} -- {global_frame_idx} frames")
            pipe.close()
        except BaseException:
            pipe.abort()
            raise

        print(f"[Renderer] Total frames rendered: {global_frame_idx}")
        outputs[orientation] = str(out_path)
        print(f"[Renderer] OK {orientation} video -> {out_path}")

    return outputs


def get_subtitle_entry_at(subtitles: list[dict], time_ms: int) -> dict | None:
    for sub in subtitles:
        if sub["start_ms"] <= time_ms <= sub["end_ms"]:
            return sub
    return None


def get_subtitle_at(subtitles: list[dict], time_ms: int) -> str:
    entry = get_subtitle_entry_at(subtitles, time_ms)
    return entry["text"] if entry else ""


def _active_word_index(entry: dict, time_ms: int) -> int:
    """Index of the word being spoken at time_ms (-1 if this entry has no word timings)."""
    words = entry.get("words")
    if not words:
        return -1
    idx = 0
    for k, w in enumerate(words):
        if w["s"] <= time_ms:
            idx = k
        else:
            break
    return idx


class _FFmpegPipe:
    """Stream raw RGB frames into one FFmpeg process (no temp JPEG files, one encode pass)."""

    def __init__(self, W: int, H: int, fps: int, audio_path: str, out_path: Path, config: dict):
        encoder = config.get("video", {}).get("encoder", "libx264")
        if encoder == "h264_nvenc":
            vargs = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-b:v", "0"]
        else:
            vargs = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps),
            "-i", "-",
            "-i", str(audio_path),
            *vargs,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            # Normalize to 44100Hz stereo -- prevents noise when concat with asset clips
            "-ar", "44100", "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self.out_path = out_path
        self.log_path = out_path.with_suffix(".ffmpeg.log")
        self._log = open(self.log_path, "wb")
        print(f"[FFmpeg] Encoding (streaming): {out_path.name}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)

    def write(self, img: Image.Image) -> None:
        if img.mode != "RGB":
            img = img.convert("RGB")
        try:
            self.proc.stdin.write(img.tobytes())
        except (BrokenPipeError, OSError):
            self._fail("FFmpeg closed its input early")

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        self._log.close()
        if rc != 0:
            self._fail(f"FFmpeg encoding failed: {rc}")
        self.log_path.unlink(missing_ok=True)
        print(f"[FFmpeg] Encoding complete: {self.out_path}")

    def abort(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass

    def _fail(self, msg: str):
        self.abort()
        try:
            tail = self.log_path.read_text(errors="replace")[-2000:]
        except Exception:
            tail = ""
        print(f"[FFmpeg] STDERR (tail):\n{tail}")
        raise RuntimeError(msg)


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