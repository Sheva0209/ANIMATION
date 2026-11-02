"""
tts_test.py — test TTS langsung tanpa pipeline
Run: D:\animasi\TTS\chatterbox-env\Scripts\python.exe D:\animasi\tts_test.py
Output: D:\animasi\audio\tts_test_output.wav
"""
import os, sys, pathlib

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

os.environ["HF_HOME"]               = r"D:\hf_cache"
os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_cache\hub"

import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS

LOCAL_MODEL = str(BASE / "TTS" / "chatterbox-model")
VOICE_REF   = str(BASE / "sound reference" / "voice_sample.wav")
OUT_WAV     = str(BASE / "audio" / "tts_test_output.wav")
TEXT = "Yo, ever wondered what it feels like to be hypnotized? [gasp] Buckle up, it's actually lowkey wild."

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Test] Device : {device}")
print(f"[Test] Model  : {LOCAL_MODEL}")
print(f"[Test] Voice  : {VOICE_REF}")

print("[Test] Loading model dari lokal...")
model = ChatterboxTurboTTS.from_local(LOCAL_MODEL, device=device)
print("[Test] Model loaded. Generating audio...")

wav = model.generate(
    text=TEXT,
    audio_prompt_path=VOICE_REF,
    top_k=1000,
    repetition_penalty=1.1,
    temperature=0.7,
    top_p=0.95,
    norm_loudness=True,
)

torchaudio.save(OUT_WAV, wav, model.sr)
print(f"[Test] DONE! Saved to: {OUT_WAV}")
print(f"[Test] Duration: {wav.shape[-1] / model.sr:.2f}s | SR: {model.sr}")
print(f"[Test] Coba play file: {OUT_WAV}")
