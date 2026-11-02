"""
tts_generator.py
Chatterbox TurboTTS -> voice-over audio per scene + merged full audio

v2.0 changes:
  - Per-scene emotion parameter control (temperature, top_p, exaggeration, cfg_weight)
  - voice_sample.wav as voice reference (WAV -- no conversion needed)
  - Auto-detect Chatterbox Turbo support for exaggeration/cfg_weight params
"""

import os
import re
import time
import yaml
import inspect
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment

# ── Setup HuggingFace cache ke drive D SEBELUM import HF libraries ─────────────
# Ini harus dilakukan sebelum chatterbox/huggingface_hub diimport!
_BASE = Path(__file__).parent.parent
load_dotenv(_BASE / ".env")  # load .env dulu biar HF_HOME terbaca
_hf_home = os.environ.get("HF_HOME", str(_BASE / "TTS" / "hf-cache"))
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(_hf_home) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(_hf_home) / "transformers"))
# Mirror HF biar gak diblokir ISP (bisa override via .env HF_ENDPOINT)
_hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _hf_endpoint
print(f"[TTS] HF cache  -> {os.environ['HF_HOME']}")
print(f"[TTS] HF mirror -> {os.environ['HF_ENDPOINT']}")

BASE_DIR = _BASE  # alias, same as _BASE defined above

# ── sys.path injection sudah tidak diperlukan ──────────────────────────────
# app.py sekarang dijalankan dari chatterbox-env Python secara langsung,
# jadi semua package (chatterbox, torch, dll) sudah tersedia di sys.path.
# Injection manual menyebabkan cffi version conflict kalau dijalankan dari
# Python lain.

# Global model cache (load once per process) -- lazy-loaded on first TTS call
_model = None
_model_lock = None  # threading.Lock — dibuat lazy supaya tidak crash saat import
_turbo_supports_emotion = None  # cached check: does Turbo support exaggeration/cfg_weight?


def _get_device() -> str:
    """Detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model():
    """Load (or return cached) ChatterboxTurboTTS model — thread-safe."""
    global _model, _model_lock
    import threading

    # Buat lock pertama kali (lazy) — aman karena GIL menjamin assignment atomik
    if _model_lock is None:
        _model_lock = threading.Lock()

    # Double-checked locking untuk thread safety
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:  # re-check setelah dapat lock
            return _model

        import torch
        import torchaudio  # noqa -- ensure torchaudio is available
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        device = _get_device()

        # ── Coba load dari folder lokal dulu (tanpa internet) ────────────────
        local_model_dir = BASE_DIR / "TTS" / "chatterbox-model"
        if local_model_dir.exists() and any(local_model_dir.glob("*.safetensors")):
            print(f"[TTS] Loading ChatterboxTurboTTS dari lokal: {local_model_dir}")
            _model = ChatterboxTurboTTS.from_local(str(local_model_dir), device=device)
        else:
            # Fallback: download dari HuggingFace Hub (butuh internet / VPN)
            print(f"[TTS] Loading ChatterboxTurboTTS on {device} (download dari HF Hub)...")
            print(f"[TTS] Tip: download manual ke {local_model_dir} biar offline!")
            _model = ChatterboxTurboTTS.from_pretrained(device=device)

        print("[TTS] Model loaded.")
    return _model


def _check_turbo_emotion_support(model) -> bool:
    """
    Auto-detect if ChatterboxTurboTTS.generate() supports
    exaggeration and cfg_weight parameters.
    Returns True if supported, False if fallback (temperature/top_p only).
    """
    global _turbo_supports_emotion
    if _turbo_supports_emotion is not None:
        return _turbo_supports_emotion

    try:
        sig = inspect.signature(model.generate)
        params = sig.parameters
        _turbo_supports_emotion = ("exaggeration" in params and "cfg_weight" in params)
    except Exception:
        _turbo_supports_emotion = False

    if _turbo_supports_emotion:
        print("[TTS] OK Chatterbox Turbo supports exaggeration + cfg_weight (full emotion control)")
    else:
        print("[TTS] WARN Chatterbox Turbo does NOT support exaggeration/cfg_weight -- using temperature/top_p fallback")

    return _turbo_supports_emotion


def _prepare_voice_prompt(audio_path: str | Path) -> str:
    """
    Chatterbox needs a WAV file for the audio prompt.
    If the provided file is already WAV, return it directly.
    Otherwise convert to a temporary WAV file and return its path.
    """
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        # Already WAV -- use directly (no conversion needed for voice_sample.wav)
        return str(audio_path)

    # Convert to WAV in a cache location (only for non-WAV files like MP3)
    wav_cache = BASE_DIR / "audio" / "voice_prompt_cache"
    wav_cache.mkdir(parents=True, exist_ok=True)
    wav_path = wav_cache / (audio_path.stem + ".wav")

    if not wav_path.exists():
        print(f"[TTS] Converting voice prompt {audio_path.name} -> WAV...")
        seg = AudioSegment.from_file(str(audio_path))
        # Chatterbox works best with 22050 or 44100 Hz mono
        seg = seg.set_channels(1).set_frame_rate(44100)
        seg.export(str(wav_path), format="wav")
        print(f"[TTS] Voice prompt cached -> {wav_path}")

    return str(wav_path)


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_narration(text: str) -> str:
    """
    Make narration sound more natural for Chatterbox Turbo:
    - Commas -> ',' + short pause (ellipsis trick) for breathing room
    - Em-dash / semicolon -> longer pause
    - Ensure sentence-final punctuation for proper cadence
    The model natively reads paralinguistic tags like [sigh], [chuckle], etc.
    so those pass through unchanged.
    """
    import re as _re
    # Comma: add a soft pause marker (three dots read as natural hesitation)
    text = _re.sub(r",\s+", ", ... ", text)
    # Em-dash or semicolon: treat as a beat pause
    # FIXED: dulu [--–] adalah Unicode range yg salah dan bisa match karakter tak terduga.
    # Sekarang pakai alternation eksplisit: '--' (double hyphen), '–' (en-dash),
    # '—' (em-dash), dan ';' masing-masing di-match secara literal.
    text = _re.sub(r"\s*(?:--|[–—;])\s*", " ... ", text)
    # Ensure text ends with sentence-final punctuation
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _get_emotion_kwargs(tts_emotion: str | None, emotion_params: dict, supports_emotion: bool) -> dict:
    """
    Build the kwargs dict for Chatterbox model.generate() based on tts_emotion.

    If Chatterbox Turbo supports exaggeration/cfg_weight, uses those.
    Otherwise falls back to temperature/top_p variation only.
    """
    if not tts_emotion or tts_emotion not in emotion_params:
        tts_emotion = "neutral"

    ep = emotion_params.get(tts_emotion, emotion_params.get("neutral", {}))

    kwargs = {
        "temperature": float(ep.get("temperature", 0.7)),
        "top_p":       float(ep.get("top_p", 0.95)),
    }

    if supports_emotion:
        kwargs["exaggeration"] = float(ep.get("exaggeration", 0.5))
        kwargs["cfg_weight"]   = float(ep.get("cfg_weight", 0.5))

    return kwargs


def generate_tts(script: dict, config: dict | None = None, safe_name: str | None = None) -> dict:
    """
    Generate TTS audio for each scene narration using ChatterboxTurboTTS.

    v2.0: Reads tts_emotion per scene and applies emotion-specific TTS parameters.

    Returns dict with:
      - full_audio: Path to merged audio file
      - scene_audio: list of {scene_id, path, duration_ms, narration, tts_emotion}
      - scene_offsets: list of start timestamps in ms
      - total_duration_ms: total merged audio length
    """
    if config is None:
        config = load_config()

    tts_cfg = config.get("tts", {})

    # ── Voice prompt (voice_sample.wav or other reference) ────────────────────
    voice_prompt_raw = tts_cfg.get("voice_prompt_path")
    voice_prompt_wav: str | None = None
    if voice_prompt_raw:
        voice_prompt_path = Path(voice_prompt_raw)
        if not voice_prompt_path.is_absolute():
            voice_prompt_path = BASE_DIR / voice_prompt_path
        if voice_prompt_path.exists():
            voice_prompt_wav = _prepare_voice_prompt(voice_prompt_path)
            print(f"[TTS] Voice reference: {voice_prompt_path.name}")
        else:
            print(f"[TTS] WARN Voice prompt not found: {voice_prompt_path} -- using default voice.")

    # ── Global fallback params ────────────────────────────────────────────────
    global_top_k          = int(tts_cfg.get("top_k", 50))
    repetition_penalty    = float(tts_cfg.get("repetition_penalty", 1.3))
    norm_loudness         = bool(tts_cfg.get("norm_loudness", True))

    # ── Emotion parameter table (from config.yaml tts.emotion_params) ─────────
    emotion_params: dict = tts_cfg.get("emotion_params", {})
    if not emotion_params:
        # Fallback defaults if config is missing emotion_params section
        emotion_params = {
            "neutral":  {"temperature": 0.70, "top_p": 0.95, "exaggeration": 0.50, "cfg_weight": 0.50},
            "excited":  {"temperature": 0.90, "top_p": 0.98, "exaggeration": 0.80, "cfg_weight": 0.40},
            "serious":  {"temperature": 0.50, "top_p": 0.90, "exaggeration": 0.30, "cfg_weight": 0.60},
            "sad":      {"temperature": 0.60, "top_p": 0.92, "exaggeration": 0.40, "cfg_weight": 0.70},
            "angry":    {"temperature": 0.85, "top_p": 0.97, "exaggeration": 0.90, "cfg_weight": 0.30},
            "confused": {"temperature": 0.75, "top_p": 0.95, "exaggeration": 0.50, "cfg_weight": 0.50},
            "happy":    {"temperature": 0.85, "top_p": 0.97, "exaggeration": 0.70, "cfg_weight": 0.40},
            "relatable":{"temperature": 0.80, "top_p": 0.96, "exaggeration": 0.60, "cfg_weight": 0.50},
            "warning":  {"temperature": 0.70, "top_p": 0.95, "exaggeration": 0.70, "cfg_weight": 0.40},
        }

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _load_model()
    supports_emotion = _check_turbo_emotion_support(model)

    # ── Output dirs ───────────────────────────────────────────────────────────
    temp_dir = BASE_DIR / "audio" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if safe_name is None:
        safe_name = re.sub(r"[^\w\s-]", "", script["title"]).strip().replace(" ", "_").lower()[:60]

    scene_results     = []
    scene_offsets     = []
    current_offset_ms = 0

    total_scenes = len(script["scenes"])
    for i, scene in enumerate(script["scenes"]):
        scene_id    = scene["id"]
        narration   = scene["narration"].strip()
        tts_emotion = scene.get("tts_emotion", "neutral")

        # Save as WAV first (torchaudio output), then convert to MP3 via pydub
        wav_path = temp_dir / f"{safe_name}_scene_{scene_id:02d}.wav"
        mp3_path = temp_dir / f"{safe_name}_scene_{scene_id:02d}.mp3"

        print(f"[TTS] Scene {scene_id}/{total_scenes} [{tts_emotion}]: generating audio...")

        # ── Preprocess: inject natural pauses at commas/dashes ────────────────
        processed = preprocess_narration(narration)

        # ── Build emotion-specific kwargs ──────────────────────────────────────
        emotion_kwargs = _get_emotion_kwargs(tts_emotion, emotion_params, supports_emotion)

        kwargs: dict = dict(
            text=processed,
            top_k=global_top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            **emotion_kwargs,
        )
        if voice_prompt_wav:
            kwargs["audio_prompt_path"] = voice_prompt_wav

        # ── Generate ───────────────────────────────────────────────────────────
        wav_tensor = model.generate(**kwargs)
        import torchaudio as ta
        ta.save(str(wav_path), wav_tensor, model.sr)

        # ── Convert WAV -> MP3 (for consistency with rest of pipeline) ─────────
        seg = AudioSegment.from_wav(str(wav_path))
        seg.export(str(mp3_path), format="mp3", bitrate="128k")
        wav_path.unlink(missing_ok=True)   # clean up temp WAV

        duration_ms = len(seg)
        scene_results.append({
            "scene_id":    scene_id,
            "path":        str(mp3_path),
            "duration_ms": duration_ms,
            "narration":   narration,
            "tts_emotion": tts_emotion,
        })
        scene_offsets.append(current_offset_ms)
        current_offset_ms += duration_ms

        print(f"[TTS] Scene {scene_id}: {duration_ms / 1000:.2f}s [{tts_emotion}] -> {mp3_path.name}")

    # ── Merge all scenes -> full audio ─────────────────────────────────────────
    print("[TTS] Merging all scenes into full audio...")
    full_audio = AudioSegment.empty()
    for result in scene_results:
        full_audio += AudioSegment.from_mp3(result["path"])

    full_audio_path = BASE_DIR / "audio" / f"{safe_name}_full.mp3"
    full_audio.export(str(full_audio_path), format="mp3", bitrate="128k")
    print(f"[TTS] Full audio saved -> {full_audio_path} ({len(full_audio)/1000:.1f}s total)")

    return {
        "full_audio":        str(full_audio_path),
        "scene_audio":       scene_results,
        "scene_offsets":     scene_offsets,
        "total_duration_ms": current_offset_ms,
    }


if __name__ == "__main__":
    import json
    import sys
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    script_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not script_path:
        print("Usage: python tts_generator.py <path_to_script.json>")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    cfg = load_config()
    result = generate_tts(script, cfg)
    print(json.dumps(result, indent=2))
