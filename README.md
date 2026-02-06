# Balauru's TTS Studio

A unified Gradio-based text-to-speech application supporting multiple TTS models with shared voice management, auto-transcription, and AI-powered post-processing.

Models run in isolated conda environments (required due to incompatible dependency versions) and communicate via local FastAPI servers. Only one TTS model is loaded at a time to stay within VRAM limits.

## Features

- **Multi-model support** — Chatterbox Turbo and Qwen3-TTS Base out of the box, with a plugin architecture for adding more
- **Dynamic UI** — Parameter controls, text processing options, and chunking strategies are driven by each model's `config.json` — no hardcoded model-specific UI
- **Voice management** — Simple folder of `.wav` files with optional `.txt` transcripts
- **Auto-transcription** — One-click Faster-Whisper transcription of voice samples
- **Text processing** — Number normalization, abbreviation expansion, punctuation cleanup, `[pause:Xs]` tags, and multiple chunking strategies
- **Post-processing** — ClearerVoice neural denoising (MossFormer2) and super-resolution (24kHz → 48kHz)
- **Output management** — WAV/MP3/FLAC export, playback speed adjustment, session ZIP download, output history browser

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  GRADIO UI  (studio.py)                   │
│                  Runs in: tts_studio conda env            │
│                                                          │
│  Voice Selector    TTS Generation     Post-Process       │
│  & Transcriber     (dynamic UI from   & Output           │
│                     config.json)                         │
│                   ┌────────────────┐                     │
│                   │ Model Registry │                     │
│                   │ (subprocess    │                     │
│                   │  management)   │                     │
│                   └───────┬────────┘                     │
└───────────────────────────┼──────────────────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
              ┌─────┴─────┐  ┌──────┴───┐
              │ Chatterbox │  │ Qwen3-TTS│
              │ Server     │  │ Server   │
              │ :5001      │  │ :5002    │
              │ conda:     │  │ conda:   │
              │ cb_env     │  │ qwen_env │
              └────────────┘  └──────────┘
               ▲ subprocess    ▲ subprocess
               └─ ONLY ONE LOADED AT A TIME ─┘
```

## Project Structure

```
TTS-Studio/
├── studio.py                       # Main Gradio application
├── requirements.txt                # Base env dependencies
│
├── core/
│   ├── transcriber.py              # Faster-Whisper wrapper (lazy load/unload)
│   ├── postprocessing.py           # ClearerVoice wrapper (denoise + super-res)
│   ├── model_registry.py           # Model discovery, subprocess + HTTP lifecycle
│   ├── text_processing.py          # Chunking, normalization, pause tags
│   └── audio_utils.py              # Concat, silence, normalize, format convert
│
├── models/
│   ├── chatterbox/
│   │   ├── server.py               # FastAPI server wrapping Chatterbox Turbo
│   │   └── config.json             # Capabilities, parameters, UI schema
│   │
│   └── qwen_tts/
│       ├── server.py               # FastAPI server wrapping Qwen3-TTS Base
│       └── config.json             # Capabilities, parameters, UI schema
│
├── voices/                         # Shared voice folder
│   ├── speaker_name.wav            # Reference audio
│   └── speaker_name.txt            # Transcript (auto-generated or manual)
│
└── output/                         # Generated audio files
```

## Setup

Three conda environments are required because Chatterbox pins `transformers==4.46.3` while Qwen3-TTS requires `transformers==4.57.3`.

### 1. Base Environment (UI + transcription + post-processing)

```bash
conda create -n tts_studio python=3.10 -y
conda activate tts_studio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio requests
pip install librosa soundfile numpy pydub
pip install faster-whisper
pip install clearvoice
```

### 2. Chatterbox Environment

```bash
conda create -n cb_env python=3.10 -y
conda activate cb_env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chatterbox-tts
pip install fastapi uvicorn
```

### 3. Qwen3-TTS Environment

```bash
conda create -n qwen_env python=3.12 -y
conda activate qwen_env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install qwen-tts
pip install fastapi uvicorn
```

## Usage

```bash
conda activate tts_studio
python studio.py
```

The app opens in your browser. To generate speech:

1. **Select a model** from the dropdown — the server starts automatically
2. **Pick a voice** from the dropdown (drop `.wav` files into `voices/`)
3. **Transcribe** the voice if using Qwen3-TTS (click Transcribe or type manually)
4. **Enter text** and click **Generate Audio**

### Voice Samples

Place `.wav` files in the `voices/` directory. For best results:

- 10–30 seconds of clean speech
- Single speaker, minimal background noise
- 24kHz or higher sample rate

Qwen3-TTS requires a transcript of the voice sample for best quality (ICL mode). Click **Transcribe** to auto-generate one with Faster-Whisper, or write it manually. The transcript is saved as a `.txt` file alongside the `.wav`.

### Text Features

- **Chunking strategies** — Sentence batching (300/500 chars), sentence split, paragraph split, or no split
- **Pause tags** — Insert `[pause:1.5s]` in your text for precise silences (Chatterbox only)
- **Sound effects** — Click tag buttons to insert `[laugh]`, `[sigh]`, `[cough]`, etc. (Chatterbox only)
- **Text cleaning** — Normalize whitespace, expand abbreviations (mr → mister), convert numbers to words, fix punctuation placement

### Post-Processing

Both options use ClearerVoice-Studio's MossFormer2 models (~200MB VRAM each):

- **Denoise** — Remove background noise, hiss, and TTS artifacts (MossFormer2_SE_48K)
- **Super-Resolution** — Upsample to 48kHz and restore high frequencies (MossFormer2_SR_48K)

## Model Parameters

### Chatterbox Turbo

| Parameter | Default | Range |
|-----------|---------|-------|
| Temperature | 0.8 | 0.05–2.0 |
| Top P | 0.95 | 0.0–1.0 |
| Top K | 1000 | 0–1000 |
| Repetition Penalty | 1.2 | 1.0–2.0 |
| Min P | 0.0 | 0.0–1.0 |
| Normalize Loudness | On | — |

### Qwen3-TTS Base

| Parameter | Default | Range |
|-----------|---------|-------|
| Temperature | 0.9 | 0.1–2.0 |
| Top K | 50 | 1–200 |
| Top P | 1.0 | 0.0–1.0 |
| Repetition Penalty | 1.05 | 1.0–2.0 |
| Language | Auto | 10 languages |
| X-Vector Only | Off | — |

**X-Vector only mode** skips the transcript requirement and uses only the speaker embedding. Quality is lower but works without transcription.

**Supported languages:** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## VRAM Management

Designed for GPUs with 8–12GB VRAM (tested on RTX 4070).

| State | VRAM Usage |
|-------|-----------|
| No model loaded | ~0 GB |
| Faster-Whisper (transcription) | ~1–2 GB (freed after use) |
| Chatterbox Turbo | ~6 GB |
| Qwen3-TTS Base (bfloat16) | ~4 GB |
| ClearerVoice (per model) | ~200 MB |

Only one TTS model occupies VRAM at a time. Switching models automatically unloads the current one. Faster-Whisper unloads immediately after transcription. A configurable cleanup interval flushes VRAM every N chunks during long generations.

## Adding New Models

To add a new TTS model:

1. Create a conda environment with the model's dependencies
2. Create `models/{model_id}/server.py` implementing 5 endpoints:
   - `POST /load` — Load model into VRAM
   - `POST /unload` — Free VRAM
   - `POST /generate` — Generate speech, return WAV bytes
   - `GET /health` — Return `{"status": "ready" | "loading" | "unloaded"}`
   - `GET /config` — Return the model's config.json
3. Create `models/{model_id}/config.json` declaring the model's name, conda env, port, capabilities, and parameters

The Gradio UI automatically discovers and supports new models — no UI code changes needed. The `config.json` drives parameter controls, text processing toggles, event tag visibility, transcript requirements, and chunking options.

## Dependencies

| Component | Package | Environment | GPU |
|-----------|---------|-------------|-----|
| Main UI | gradio | tts_studio | No |
| Model servers | fastapi, uvicorn | cb_env, qwen_env | No |
| HTTP client | requests | tts_studio | No |
| Audio I/O | librosa, soundfile, pydub | tts_studio | No |
| Transcription | faster-whisper | tts_studio | Optional |
| Post-processing | clearvoice | tts_studio | Yes |
| Chatterbox TTS | chatterbox-tts | cb_env | Yes |
| Qwen3-TTS | qwen-tts | qwen_env | Yes |
