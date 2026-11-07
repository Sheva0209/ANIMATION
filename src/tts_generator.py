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
import json
import hashlib
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


def _tight_params(tts_cfg: dict) -> dict:
    """Dead-air trimming parameters (all overridable under `tts:` in config.yaml)."""
    return {
        "enabled":           bool(tts_cfg.get("trim_silence", True)),
        "max_pause_ms":      int(tts_cfg.get("max_pause_ms", 450)),     # longer pauses are shortened to this
        "edge_pad_ms":       int(tts_cfg.get("edge_pad_ms", 120)),      # quiet kept at scene start/end
        "silence_offset_db": float(tts_cfg.get("silence_offset_db", 16)),  # "silence" = N dB below average loudness
        "min_silence_ms":    200,
        "speech_pad_ms":     60,                                        # safety margin so soft consonants are not clipped
    }


def _tighten_audio(seg: AudioSegment, p: dict) -> AudioSegment:
    """
    Remove dead air from one scene: trim leading/trailing silence and shorten long pauses.
    Speech itself is never touched; if nothing can be detected the audio is returned unchanged.
    """
    from pydub.silence import detect_nonsilent

    if len(seg) < 600 or seg.dBFS == float("-inf"):
        return seg
    thresh = seg.dBFS - p["silence_offset_db"]       # relative threshold: robust to loudness differences
    spans = detect_nonsilent(seg, min_silence_len=p["min_silence_ms"], silence_thresh=thresh, seek_step=10)
    if not spans:
        return seg

    pad, n = p["speech_pad_ms"], len(seg)
    merged: list[list[int]] = []
    for s, e in spans:
        s, e = max(0, s - pad), min(n, e + pad)
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    edge, max_pause = p["edge_pad_ms"], p["max_pause_ms"]
    out = seg[max(0, merged[0][0] - edge):merged[0][0]]
    for i, (s, e) in enumerate(merged):
        if i > 0:
            prev_e = merged[i - 1][1]
            gap = s - prev_e
            if gap <= max_pause:
                out += seg[prev_e:s]                              # natural pause: keep as is
            else:
                half = max_pause // 2                             # long pause: keep only its edges
                out += seg[prev_e:prev_e + half].fade_out(8) + seg[s - half:s].fade_in(8)
        out += seg[s:e]
    out += seg[merged[-1][1]:merged[-1][1] + edge]
    return out


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

    # ── Output dirs ───────────────────────────────────────────────────────────
    cache_dir = BASE_DIR / "audio" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_cache = bool(tts_cfg.get("use_cache", True))
    tparams = _tight_params(tts_cfg)
    tight_tag = hashlib.sha1(json.dumps(tparams, sort_keys=True).encode()).hexdigest()[:6]

    if safe_name is None:
        safe_name = re.sub(r"[^\w\s-]", "", script["title"]).strip().replace(" ", "_").lower()[:60]

    # ── Voice reference signature (cache must change if the reference changes) ─
    voice_sig = None
    if voice_prompt_wav:
        vp = Path(voice_prompt_wav)
        st = vp.stat()
        voice_sig = f"{vp.name}:{st.st_size}:{st.st_mtime_ns}"

    # ── Plan every scene first: text + params -> cache key ────────────────────
    plan = []
    for scene in script["scenes"]:
        narration   = scene["narration"].strip()
        tts_emotion = scene.get("tts_emotion", "neutral")
        processed   = preprocess_narration(narration)
        key = _scene_cache_key(
            processed, tts_emotion, emotion_params, tts_cfg, voice_sig,
            global_top_k, repetition_penalty, norm_loudness,
        )
        plan.append({
            "scene_id":    scene["id"],
            "narration":   narration,
            "tts_emotion": tts_emotion,
            "processed":   processed,
            "wav_path":    cache_dir / f"{key}.wav",
        })

    # ── Load the model ONLY if at least one scene is not cached ───────────────
    missing = [p for p in plan if not (use_cache and p["wav_path"].exists())]
    model = None
    supports_emotion = False
    if missing:
        model = _load_model()
        supports_emotion = _check_turbo_emotion_support(model)
    else:
        print("[TTS] All scenes found in cache -- model not loaded")

    scene_results     = []
    scene_offsets     = []
    current_offset_ms = 0
    full_audio        = AudioSegment.empty()
    generated_count   = 0

    total_scenes = len(plan)
    for i, item in enumerate(plan):
        scene_id    = item["scene_id"]
        tts_emotion = item["tts_emotion"]
        wav_path    = item["wav_path"]

        if use_cache and wav_path.exists():
            print(f"[TTS] Scene {scene_id}/{total_scenes} [{tts_emotion}]: cached")
        else:
            print(f"[TTS] Scene {scene_id}/{total_scenes} [{tts_emotion}]: generating audio...")
            emotion_kwargs = _get_emotion_kwargs(tts_emotion, emotion_params, supports_emotion)
            kwargs: dict = dict(
                text=item["processed"],
                top_k=global_top_k,
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                **emotion_kwargs,
            )
            if voice_prompt_wav:
                kwargs["audio_prompt_path"] = voice_prompt_wav

            wav_tensor = model.generate(**kwargs)
            import torchaudio as ta
            # Write to a temp name first so a crash never leaves a half-written cache file
            tmp_path = wav_path.with_name(wav_path.stem + ".tmp.wav")
            ta.save(str(tmp_path), wav_tensor, model.sr)
            os.replace(tmp_path, wav_path)
            generated_count += 1

        seg = AudioSegment.from_wav(str(wav_path))
        raw_ms = len(seg)
        audio_path = wav_path
        if tparams["enabled"]:
            # The raw WAV stays untouched in the cache; the tightened copy is what the rest of the
            # pipeline uses (durations, subtitles, Whisper alignment all follow THIS file).
            tight_path = wav_path.with_name(f"{wav_path.stem}.t{tight_tag}.wav")
            if not (use_cache and tight_path.exists()):
                tmp_t = tight_path.with_name(tight_path.stem + ".tmp.wav")
                _tighten_audio(seg, tparams).export(str(tmp_t), format="wav")
                os.replace(tmp_t, tight_path)
            seg = AudioSegment.from_wav(str(tight_path))
            audio_path = tight_path
        duration_ms = len(seg)
        scene_results.append({
            "scene_id":    scene_id,
            "path":        str(audio_path),
            "duration_ms": duration_ms,
            "narration":   item["narration"],
            "tts_emotion": tts_emotion,
        })
        scene_offsets.append(current_offset_ms)
        current_offset_ms += duration_ms
        full_audio += seg

        if raw_ms - duration_ms >= 300:
            print(f"[TTS] Scene {scene_id}: {raw_ms / 1000:.2f}s -> {duration_ms / 1000:.2f}s "
                  f"(trimmed {(raw_ms - duration_ms) / 1000:.1f}s dead air) [{tts_emotion}]")
        else:
            print(f"[TTS] Scene {scene_id}: {duration_ms / 1000:.2f}s [{tts_emotion}]")

    # ── Merge all scenes -> full audio (WAV: lossless, encoded once by FFmpeg) ─
    full_audio_path = BASE_DIR / "audio" / f"{safe_name}_full.wav"
    full_audio.export(str(full_audio_path), format="wav")
    print(f"[TTS] Full audio saved -> {full_audio_path} ({len(full_audio)/1000:.1f}s total, "
          f"{generated_count} new / {total_scenes - generated_count} cached)")

    return {
        "full_audio":        str(full_audio_path),
        "scene_audio":       scene_results,
        "scene_offsets":     scene_offsets,
        "total_duration_ms": current_offset_ms,
    }


def _scene_cache_key(processed, tts_emotion, emotion_params, tts_cfg, voice_sig,
                     top_k, repetition_penalty, norm_loudness) -> str:
    """
    Hash everything that changes how ONE scene sounds: text, emotion + its params,
    global sampling params and the voice reference. Editing one scene (or one
    emotion's params) only invalidates the scenes it actually affects.
    """
    emo = tts_emotion if tts_emotion in emotion_params else "neutral"
    payload = {
        "text": processed,
        "emotion": emo,
        "emotion_params": emotion_params.get(emo, {}),
        "temperature": tts_cfg.get("temperature"),
        "top_p": tts_cfg.get("top_p"),
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "norm_loudness": norm_loudness,
        "voice": voice_sig,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


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