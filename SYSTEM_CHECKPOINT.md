# 🎬 Sheva Animation Studio — System Checkpoint

> **Tujuan dokumen ini**: Jadi referensi cepat sebelum buat perubahan apapun ke sistem.
> Baca ini dulu, baru buka file. Hemat waktu 30+ menit.

---

## 📁 Struktur Proyek

```
d:\animasi\
├── app.py                    # Flask web server + REST API + pipeline runner (UI entry point)
├── main.py                   # CLI entry point (gunakan chatterbox-env!)
├── config.yaml               # ⚙️ SATU-SATUNYA config file — ubah behavior dari sini
├── .env                      # API keys (OpenAI, ElevenLabs, Gemini, HF_HOME)
├── tts_test.py               # Test TTS langsung tanpa pipeline (debugging)
│
├── src/
│   ├── script_generator.py   # Step 1: GPT-4o → script JSON
│   ├── tts_generator.py      # Step 2: ChatterboxTurboTTS → audio MP3 per scene
│   ├── transcriber.py        # Step 3: script + TTS timing → SRT subtitle
│   ├── scene_generator.py    # (Opsional) Gemini Imagen → custom scene image
│   ├── renderer.py           # Step 4: Pillow compositor + FFmpeg → MP4
│   └── pipeline.py           # CLI pipeline orchestrator (dipanggil main.py)
│
├── poses/                    # PNG karakter Sheva (transparent bg)
├── backgrounds/              # Background images (kamar.jpg, kampus.jpg, rooftop.jpeg)
├── audio/                    # Output audio TTS + checkpoint JSON
├── subtitles/                # Output SRT files
├── output/                   # Output video MP4 (portrait/)
├── scripts/                  # Saved script JSON dari GPT-4o
├── assets/uploads/           # User-uploaded asset clips (dari UI)
│
└── TTS/
    ├── chatterbox-env/       # ⚠️ Virtual env WAJIB untuk TTS (PyTorch, Chatterbox)
    └── chatterbox-model/     # Model files lokal (*.safetensors) — offline support
```

---

## 🔄 Pipeline Flow

```
[UI/CLI] → Topic input
    │
    ▼
[script_generator.py]
  GPT-4o → JSON script
  (title, scenes[id, background, pose, narration, tts_emotion, subtitle_emotion])
    │
    ▼
[tts_generator.py]
  ChatterboxTurboTTS → MP3 per scene + full merged audio
  Voice ref: sound reference/voice_sample.wav
  Checkpoint: audio/{safe_name}_checkpoint.json
    │
    ▼
[transcriber.py]
  Script + TTS timing → .srt subtitle (NO Whisper, pure script-based)
    │
    ▼
[scene_generator.py] (opsional, jika ada generated_scene)
  Gemini Imagen / GPT-image-1 → custom scene PNG
    │
    ▼
[renderer.py]
  Pillow: per-frame compositor (bg + Ken Burns + character + subtitle)
  FFmpeg: frames + audio → MP4
  Output: output/portrait/{safe_name}.mp4
```

---

## ⚠️ CRITICAL: Python Environment

**SELALU gunakan Python dari `chatterbox-env` untuk menjalankan apapun yang berhubungan dengan TTS:**

```powershell
# Jalankan app.py (Flask UI server)
D:\animasi\TTS\chatterbox-env\Scripts\python.exe app.py

# Test TTS manual
D:\animasi\TTS\chatterbox-env\Scripts\python.exe tts_test.py

# CLI pipeline
D:\animasi\TTS\chatterbox-env\Scripts\python.exe main.py "topik video"
```

**Mengapa?** Chatterbox TTS butuh PyTorch + torchaudio yang hanya ada di `chatterbox-env`.
Kalau pakai Python sistem/lain → model mungkin gagal load atau behavior aneh.

---

## ⚙️ Config Quick Reference (`config.yaml`)

### TTS Parameters
```yaml
tts:
  voice_prompt_path: "sound reference/voice_sample.wav"  # Voice reference Sheva
  top_k: 1000               # ⚠️ JANGAN < 200 → bisa noise/garbled audio
  repetition_penalty: 1.1   # ⚠️ JANGAN > 1.2 → noise/artifacts
  norm_loudness: true        # Normalize ke -27 LUFS
  temperature: 0.7           # Override via emotion_params per scene
  top_p: 0.95
```

### Video Output
```yaml
video:
  fps: 30
  portrait: [1080, 1920]    # Saat ini hanya render portrait
  portrait_only: true        # Set false untuk render landscape juga (lebih lama)
```

### LLM Script Generation
```yaml
llm:
  model: gpt-4o
  language: en               # en | id
  niche: science_facts       # science_facts | song_meaning | education
  max_scenes: 14
  target_duration: 60        # seconds
  max_generated_scenes: 0    # 0 = disable AI scene generation (hemat biaya)
```

### Poses & Backgrounds
- **Tambah pose baru**: taruh PNG di `poses/`, tambah entry di `config.yaml → pose_map`
- **Tambah background baru**: taruh gambar di `backgrounds/`, tambah entry di `config.yaml → background_map`

---

## 🐛 Known Bugs & Fixes

### Bug #1: Dubbing Kacau via UI (FIXED — Nov 2026)
**Symptom**: TTS via UI menghasilkan audio berisi dengung tanpa kata-kata.
**Root cause**: Regex `[--–]` di `preprocess_narration()` adalah **Unicode character range**
dari `-` ke `–`, yang secara tidak sengaja match **semua karakter alfanumerik**.
Akibatnya seluruh teks narasi dihancurkan menjadi `" ... ... ... ... "`.

**Bukti**:
```
Input : "Hello world -- this is a test"
Output LAMA: " ... ... ... ... ... ... ... ... ... ..."  ← semua kata hilang!
Output BARU: "Hello world ... this is a test"             ← benar
```

**Fix** di [`src/tts_generator.py`](src/tts_generator.py) — `preprocess_narration()`:
```python
# LAMA (BUGGY) — Unicode range [--–] match semua karakter!
text = re.sub(r"\s*[--–;]\s*", " ... ", text)

# BARU (FIXED) — explicit alternation
text = re.sub(r"\s*(?:--|[–—;])\s*", " ... ", text)
```

**Mengapa hanya muncul di UI?**
`tts_test.py` (terminal) langsung panggil `model.generate(text=...)` tanpa lewat
`preprocess_narration()`. Pipeline UI memanggil `generate_tts()` → `preprocess_narration()` → teks rusak.

### Bug #2: Model Loading Thread Safety (FIXED — Nov 2026)
**Symptom**: Potensi race condition jika 2 render request datang bersamaan ke Flask.
**Fix**: Tambah `threading.Lock()` (double-checked locking) di `_load_model()`.

---

## 🏗️ Key Module Details

### `tts_generator.py` — Gotchas

| Hal | Detail |
|-----|--------|
| **Model cache** | `_model` adalah global — load sekali per process, pakai untuk semua request |
| **Checkpoint** | Disimpan di `audio/{safe_name}_checkpoint.json`. Jika file audio masih ada, TTS di-SKIP |
| **Hapus checkpoint** | `Remove-Item "d:\animasi\audio\*_checkpoint.json"` untuk force re-generate |
| **top_k < 200** | Menghasilkan noise/artifacts — jangan turunkan di bawah 200 |
| **repetition_penalty > 1.2** | Menghasilkan noise — jangan naikkan di atas 1.2 |
| **voice_sample.wav** | Di `sound reference/voice_sample.wav` — mono, 44100 Hz direkomendasikan |
| **Paralinguistic tags** | `[sigh]`, `[chuckle]`, `[gasp]`, dll — pass through langsung ke model, jangan dihapus |

### `renderer.py` — Gotchas

| Hal | Detail |
|-----|--------|
| **Ken Burns** | Slow zoom-in 0-10% per scene, berjalan via `ken_burns_crop()` |
| **Blink system** | Pose dengan `has_blink: true` di config punya `blink_pair` — frame PNG terpisah untuk mata menutup |
| **Crossfade** | Antar scene dissolve ke frame pertama scene berikutnya (BUKAN fade to black) |
| **Frame cleanup** | Folder `output/portrait/frames/{safe_name}/` otomatis dihapus setelah encode |
| **Audio sync** | Durasi setiap scene diambil dari `audio_data["scene_audio"][i]["duration_ms"]` |
| **FFmpeg output** | `-ar 44100 -ac 2` dipaksa untuk mencegah noise saat asset splice |

### `app.py` — Gotchas

| Hal | Detail |
|-----|--------|
| **Pipeline thread** | TTS + render jalan di `_run_pipeline_thread()` — background daemon thread |
| **SSE progress** | Client polling `/api/progress/{job_id}` — queue-based, timeout 30s → heartbeat |
| **Asset splice** | `_splice_assets()` pakai FFmpeg concat demuxer — semua clip dinormalisasi ke 44100Hz stereo |
| **Checkpoint load** | UI cek `*_checkpoint.json` sebelum generate TTS — hapus jika mau force re-TTS |

### `script_generator.py` — Gotchas

| Hal | Detail |
|-----|--------|
| **3 niche** | `science_facts`, `song_meaning`, `education` — masing-masing punya system prompt berbeda |
| **2 bahasa** | `en` dan `id` — pose/background name selalu English (dari config) |
| **Pose validation** | Jika LLM pakai pose tidak valid → fallback ke `"halfbody explaining"` |
| **Background validation** | Jika LLM pakai bg tidak valid → fallback ke `"kampus"` |
| **`generated_scene`** | Field di scene JSON — jika `null` pakai pose biasa, jika ada string → trigger Gemini |

### `transcriber.py` — Gotchas

| Hal | Detail |
|-----|--------|
| **No Whisper** | Subtitle dibangun dari teks narasi script + timing audio (`duration_ms`) |
| **Chunk timing** | Proporsional — `chunk_ms = (chunk_words / total_words) * total_ms` |
| **Max words per chunk** | `config.yaml → subtitle.max_words_per_chunk` (default 7) |
| **Whisper ada** | `transcribe_audio()` masih ada tapi LEGACY — jangan pakai kecuali terpaksa |

---

## 🔌 API Endpoints (Flask)

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `GET /` | GET | Serve `static/index.html` |
| `/api/config` | GET | Return poses, backgrounds, emotions untuk UI dropdowns |
| `/api/generate-script` | POST | `{topic, niche, language}` → script JSON |
| `/api/render` | POST | `{script, topic, insert_assets}` → `{job_id}` |
| `/api/progress/{job_id}` | GET | SSE stream progress (step, progress, done, error) |
| `/api/jobs/{job_id}` | GET | Status snapshot job |
| `/api/upload-asset` | POST | Upload gambar/video untuk asset splice |
| `/api/outputs/{path}` | GET | Serve output video MP4 |
| `/api/outputs-list` | GET | List semua video yang sudah di-render |

---

## 🎭 Scene JSON Schema

```json
{
  "id": 1,
  "background": "kamar",           // dari config.yaml → background_map
  "pose": "halfbody explaining",   // dari config.yaml → pose_map
  "character_position": "center",  // center | left | right
  "narration": "Okay so...",       // teks yang dibacakan + bisa ada [gasp] dll
  "mood": "neutral",               // mood deskriptif (label saja, tidak dipakai code)
  "subtitle_emotion": "excited",   // kontrol warna & style subtitle
  "tts_emotion": "excited",        // kontrol parameter TTS (temperature, exaggeration, dll)
  "generated_scene": null,         // atau string prompt untuk Gemini image generation
  "estimated_duration": 8.5,       // perkiraan detik (fallback jika TTS belum jalan)
  "custom_background_path": null   // path ke custom bg (dari UI upload, opsional)
}
```

**`tts_emotion` → TTS parameter mapping** (dari `config.yaml → tts.emotion_params`):
| Emotion | temperature | exaggeration | cfg_weight |
|---------|-------------|--------------|------------|
| neutral | 0.70 | 0.50 | 0.50 |
| excited | 0.90 | 0.80 | 0.40 |
| serious | 0.50 | 0.30 | 0.60 |
| angry | 0.85 | 0.90 | 0.30 |
| sad | 0.60 | 0.40 | 0.70 |
| happy | 0.85 | 0.70 | 0.40 |
| relatable | 0.80 | 0.60 | 0.50 |
| warning | 0.70 | 0.70 | 0.40 |

---

## 🚀 Common Tasks

### Restart server setelah ubah code
```powershell
# Kill server lama dulu (Ctrl+C di terminal), lalu:
D:\animasi\TTS\chatterbox-env\Scripts\python.exe app.py
```

### Force re-generate TTS (hapus checkpoint)
```powershell
Remove-Item "D:\animasi\audio\*_checkpoint.json" -Force
```

### Test TTS langsung (bypass pipeline)
```powershell
D:\animasi\TTS\chatterbox-env\Scripts\python.exe tts_test.py
# Output: D:\animasi\audio\tts_test_output.wav
```

### Generate script via CLI
```powershell
D:\animasi\TTS\chatterbox-env\Scripts\python.exe src\script_generator.py "topik video"
```

### Render via CLI (lengkap)
```powershell
D:\animasi\TTS\chatterbox-env\Scripts\python.exe main.py "topik video"
```

### Tambah pose baru
1. Taruh PNG (transparent bg) ke `d:\animasi\poses\`
2. Tambah entry di `config.yaml → pose_map`:
   ```yaml
   "nama_pose_baru":
     type: halfbody          # fullbody | halfbody | halfbody_cropped
     mood: [keyword, ...]    # LLM pilih pose berdasarkan keyword ini
     has_blink: false
   ```
3. Restart server

### Tambah background baru
1. Taruh gambar ke `d:\animasi\backgrounds\`
2. Tambah di `config.yaml → background_map`:
   ```yaml
   nama_bg:
     file: "nama_file.jpg"
     vibe: [keyword, ...]
   ```
3. Restart server

---

## 🔑 API Keys (`.env`)

| Key | Dipakai untuk |
|-----|--------------|
| `OPENAI_API_KEY` | GPT-4o script generation + Whisper (legacy) + gpt-image-1 fallback |
| `GEMINI_API_KEY` | Gemini Imagen scene generation (opsional, hanya jika `generated_scene` ≠ null) |
| `ELEVENLABS_API_KEY` | Tidak aktif dipakai saat ini |
| `HF_HOME` | Cache HuggingFace ke `D:\hf_cache` (biar C: gak penuh) |
| `HF_ENDPOINT` | `https://hf-mirror.com` — bypass ISP block untuk HF download |

---

## 📝 Changelog

| Tanggal | Perubahan |
|---------|-----------|
| Nov 2026 | **Fix bug regex `preprocess_narration`** — `[--–]` diganti `(?:\|\|[–—;])` — dubbing via UI sekarang normal |
| Nov 2026 | **Thread-safe model loading** — tambah `threading.Lock()` di `_load_model()` |
| Nov 2026 | Subtitle system diganti dari Whisper → script-based (`generate_srt_from_script`) |
| Nov 2026 | Asset splice feature — insert gambar/video di antara scene via UI |
| Nov 2026 | Ken Burns effect + crossfade dissolve antar scene |
| Nov 2026 | Per-scene `tts_emotion` parameter control |
| Nov 2026 | `portrait_only: true` untuk mempercepat render |
