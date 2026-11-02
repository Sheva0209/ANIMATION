# 🎬 SHEVA ANIMATION STUDIO — UPGRADE PLAN v2.0
> **Last updated:** 2026-11-02  
> **Status:** ✅ COMPLETE

---

## 📌 RINGKASAN UPGRADE

| # | Area | Status |
|---|------|--------|
| 1 | **TTS Chatterbox + Emotion Control** — per-scene `tts_emotion` → Chatterbox params | ✅ DONE |
| 2 | **Voice Reference `voice_sample.wav`** — WAV langsung dipakai tanpa konversi | ✅ DONE |
| 3 | **Niche: Science/Teori** — `_prompt_science_en()` + routing `science_facts` | ✅ DONE |
| 4 | **UI Overhaul** — niche selector, TTS emotion per scene, keyboard shortcuts, live word count | ✅ DONE |

---

## 🗺️ FILE MAP SISTEM (v2.0)

```
d:\animasi\
├── app.py                    ← Flask REST API + SSE | niche/lang override per-request
├── main.py                   ← CLI entry point
├── config.yaml               ← niche: science_facts | voice: voice_sample.wav | emotion_params
├── .env                      ← API keys (OPENAI_API_KEY)
├── PLANNING.md               ← ← ini file ini
│
├── src/
│   ├── script_generator.py   ← LLM → JSON script | +_prompt_science_en() | +tts_emotion schema
│   ├── tts_generator.py      ← ChatterboxTurboTTS | per-scene emotion params | auto-detect exaggeration
│   ├── transcriber.py        ← Script → SRT subtitle
│   ├── renderer.py           ← Pillow + FFmpeg → video
│   ├── pipeline.py           ← Orchestrator (step 1-4)
│   └── scene_generator.py    ← Gemini Imagen (AI-generated scenes)
│
├── static/index.html         ← Web UI v2.0 | niche selector | TTS emotion | keyboard shortcuts
├── sound reference/
│   ├── voice_sample.wav      ← ✅ Active voice reference (Sheva)
│   └── SAMPLE _GWEN.mp3      ← Tidak dipakai
│
├── scripts/                  ← Generated script JSON (+ tts_emotion field)
├── audio/                    ← Generated audio
│   ├── temp/                 ← Scene MP3 temp
│   └── voice_prompt_cache/   ← Cache WAV (kosong jika pakai WAV langsung)
├── output/
│   ├── portrait/             ← Output 9:16
│   └── landscape/            ← Output 16:9
├── poses/                    ← Sprite Sheva
└── backgrounds/              ← BG images
```

---

## 🎭 UPGRADE 1: TTS Emotion Control — DONE ✅

### Implementasi
- **`tts_generator.py`** — fully rewritten:
  - `_check_turbo_emotion_support()` — auto-detect apakah Chatterbox Turbo support `exaggeration`/`cfg_weight` (cek 1x saat startup via `inspect.signature`)
  - `_get_emotion_kwargs()` — build per-scene kwargs dari `tts_emotion` field
  - Log tiap scene: `[TTS] Scene 3/8 [excited]: generating...`
- **`config.yaml`** — tambah `tts.emotion_params` (9 emosi × 4 param)

### Emotion → Parameter Mapping (implemented)

| `tts_emotion` | `temperature` | `exaggeration` | `cfg_weight` | Karakter |
|---|---|---|---|---|
| `neutral` | 0.70 | 0.50 | 0.50 | Normal, stabil |
| `excited` | 0.90 | 0.80 | 0.40 | Energik, cepat |
| `serious` | 0.50 | 0.30 | 0.60 | Datar, berwibawa |
| `sad` | 0.60 | 0.40 | 0.70 | Lembut, dalam |
| `angry` | 0.85 | 0.90 | 0.30 | Keras, intens |
| `confused` | 0.75 | 0.50 | 0.50 | Agak ragu |
| `happy` | 0.85 | 0.70 | 0.40 | Ceria, ringan |
| `relatable` | 0.80 | 0.60 | 0.50 | Casual, santai |
| `warning` | 0.70 | 0.70 | 0.40 | Tegas, waspada |

> **Note:** Jika ChatterboxTurboTTS tidak support `exaggeration`/`cfg_weight`, fallback otomatis ke `temperature` + `top_p` saja. Auto-detected saat pertama kali model di-load.

---

## 🎙️ UPGRADE 2: Voice Reference — DONE ✅

- **`config.yaml`:** `voice_prompt_path: "sound reference/voice_sample.wav"`
- File `.wav` → langsung dipakai tanpa konversi (bypass MP3→WAV cache)
- `_prepare_voice_prompt()` di `tts_generator.py` sudah handle WAV shortcut

---

## 🔬 UPGRADE 3: Niche Science/Teori — DONE ✅

### Prompt Sheva untuk Science
- Tone: curious, amazed, blunt, relatable. BUKAN akademis.
- Catchphrases: *"okay this is actually insane"*, *"scientists literally proved that..."*, *"no one talks about this enough"*, *"this breaks my brain a little"*, *"no cap"*, *"lowkey wild"*
- Tags: `[gasp]`, `[chuckle]`, `[sigh]`, `[clear throat]`
- Konten: black holes, quantum, dark matter, Fermi paradox, simulation theory, dll

### Files Updated
- `src/script_generator.py` → `_prompt_science_en()`, routing `science_facts`
- `config.yaml` → `llm.niche: science_facts`
- `app.py` → endpoint support `niche` override dari UI (per-request, tanpa restart)
- `static/index.html` → `science_facts` niche option

---

## 🖥️ UPGRADE 4: UI Overhaul — DONE ✅

### Features Implemented

| Feature | Status |
|---------|--------|
| Niche selector (Science / Song / Education) | ✅ |
| Language selector (EN / ID) | ✅ |
| Voice reference indicator (voice_sample.wav) | ✅ |
| Dynamic suggestions per niche | ✅ |
| Universal topic input (+ song-extras for song niche) | ✅ |
| Scene editor 4-kolom (Pose/BG/TTS Emotion/Sub Emotion) | ✅ |
| TTS Emotion dropdown per scene (berwarna) | ✅ |
| Emotion badge berwarna per scene | ✅ |
| Live word count + estimated duration | ✅ |
| Auto-save script ke localStorage | ✅ |
| localStorage persistence (niche/lang) | ✅ |
| Keyboard shortcuts | ✅ |
| Inline video player (step-done) | ✅ |
| History dengan video hover-preview | ✅ |
| Label "Chatterbox TTS" (bukan ElevenLabs) | ✅ |
| Label "from script" (bukan Whisper) | ✅ |

### Keyboard Shortcuts
| Shortcut | Aksi |
|---|---|
| `Ctrl+Enter` | Generate Script (di step input) |
| `Ctrl+R` | Render Video (di step review) |
| `Ctrl+Shift+N` | New Video |
| `Escape` | Back to prompt (dari review) |

---

## 📋 CHECKPOINT TRACKER — FINAL

### ✅ Pre-v2.0 (existing)
- [x] ChatterboxTurboTTS terintegrasi di `tts_generator.py`
- [x] Voice prompt support (MP3 → WAV conversion cache)
- [x] SSE progress streaming di `app.py`
- [x] Script JSON schema dengan `subtitle_emotion` per scene
- [x] Paralinguistic tags `[sigh]`, `[chuckle]`, dll ada di LLM prompt
- [x] Emotion-based subtitle styling di `config.yaml`
- [x] Asset splicing (insert image/video between scenes)
- [x] Portrait-only render mode
- [x] TTS checkpoint/resume system
- [x] Flask REST API + Web UI

### ✅ v2.0 Phase 1 — Backend (2026-11-02)
- [x] **[VOICE-1]** `voice_prompt_path` → `voice_sample.wav`
- [x] **[TTS-1]** Auto-detect `exaggeration`/`cfg_weight` support via `inspect.signature`
- [x] **[TTS-2]** `emotion_params` mapping (9 emosi) di `config.yaml`
- [x] **[TTS-3]** `tts_generator.py` — per-scene emotion kwargs
- [x] **[SCRIPT-1]** `tts_emotion` field di semua 3 prompt JSON schemas
- [x] **[SCRIPT-2]** `_prompt_science_en()` function
- [x] **[SCRIPT-3]** `build_system_prompt()` routing science_facts
- [x] **[CONFIG-1]** `config.yaml` → `niche: science_facts`
- [x] **[API-1]** `app.py` → niche/language override per-request

### ✅ v2.0 Phase 2 — UI (2026-11-02)
- [x] **[UI-1]** TTS Emotion dropdown per scene
- [x] **[UI-2]** Emotion badge berwarna (TTS + subtitle)
- [x] **[UI-3]** Niche selector + hero update dynamic
- [x] **[UI-4]** Voice reference indicator
- [x] **[UI-5]** Inline video player (step-done) + history hover-preview
- [x] **[UI-6]** Scene editor 4-kolom
- [x] **[UI-7]** Live word count + estimated duration
- [x] **[UI-8]** Auto-save ke localStorage
- [x] **[UI-SHORTCUTS]** Keyboard shortcuts (Ctrl+Enter, Ctrl+R, Ctrl+Shift+N, Escape)
- [x] **[POLISH-1]** Labels updated (Chatterbox TTS, from script)
- [x] **[POLISH-2]** `newVideo()` clear semua input + re-init niche UI

---

## 🔧 TECHNICAL NOTES

### Script JSON Schema v2.0
```json
{
  "id": 1,
  "background": "kamar",
  "pose": "halfbody explaining",
  "character_position": "center",
  "narration": "[gasp] okay this is actually insane...",
  "mood": "excited",
  "subtitle_emotion": "excited",
  "tts_emotion": "excited",
  "generated_scene": null,
  "estimated_duration": 8.5
}
```

### Niche Values
| Value | Prompt Function | User Label |
|-------|----------------|------------|
| `science_facts` | `_prompt_science_en()` | 🔬 Science & Theories |
| `song_meaning` | `_prompt_song_meaning_en()` | 🎵 Song Meaning |
| `education` | `_prompt_education_en()` / `_id()` | 📚 Education / Tips |

### Voice Reference Flow
```
config.yaml: voice_prompt_path: "sound reference/voice_sample.wav"
  ↓
_prepare_voice_prompt() → suffix == ".wav" → return as-is (no conversion)
  ↓
model.generate(audio_prompt_path=wav_path, ...)
```

---

## 🚀 CARA MENJALANKAN

```bash
# Web UI
python app.py
# → http://localhost:5000

# CLI
python main.py "kenapa waktu melambat di dekat lubang hitam?"
python main.py "what is quantum entanglement" --skip-tts --audio audio/existing.mp3
```

---

## 📝 NEXT IDEAS (future)

- [ ] Background music/ambience layer option
- [ ] Multiple character support (Sheva + teman)
- [ ] Batch generation (multiple topics sekaligus)
- [ ] Video thumbnail auto-generator
- [ ] TikTok/Reels direct upload integration
- [ ] Fine-tuning voice reference dengan lebih banyak sample
