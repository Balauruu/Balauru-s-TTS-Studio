# Balauru's TTS Studio — Final Implementation Plan

## 1. Project Overview

A unified Gradio-based TTS application supporting multiple TTS models (starting with Chatterbox Turbo and Qwen3-TTS Base) with shared voice management, auto-transcription, and AI-powered post-processing. Models run in isolated conda environments (required due to incompatible `transformers` versions) and communicate via local FastAPI servers. Only one TTS model is loaded at a time (load-on-demand) to stay within RTX 4070 VRAM limits.

**Existing code to port:** The Chatterbox Gradio UI (`D:\AI\chatterbox\my_gradio_tts_turbo.py`, 576 lines) and headless batch script (`my_tts_turbo.py`, 169 lines) contain battle-tested text preprocessing, chunking, pause tag parsing, and generation logic that will be ported into this project.

---

## 2. Architecture

### 2.1 High-Level Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     GRADIO UI  (studio.py)                       │
│                     Runs in: tts_studio conda env                │
│                                                                  │
│  ┌─────────────────┐  ┌────────────────────┐  ┌──────────────┐  │
│  │ Voice Selector  │  │ TTS Generation     │  │ Post-Process  │  │
│  │ & Transcriber   │  │ (dynamic UI from   │  │ & Output      │  │
│  │                 │  │  config.json)      │  │               │  │
│  └────────┬────────┘  └─────────┬──────────┘  └──────┬───────┘  │
│           │           ┌─────────▼──────────┐         │          │
│           │           │  Model Registry    │         │          │
│           │           │  (subprocess mgmt, │         │          │
│           │           │   HTTP lifecycle)  │         │          │
│           │           └─────────┬──────────┘         │          │
└───────────┼─────────────────────┼────────────────────┼──────────┘
            │                     │                    │
            │             ┌───────┴────────┐           │
            │             │                │           │
   ┌────────▼─────┐  ┌────▼─────┐  ┌──────▼──┐  ┌────▼──────────┐
   │ Faster-      │  │ CB       │  │ Qwen    │  │ ClearerVoice  │
   │ Whisper      │  │ Server   │  │ Server  │  │ (in-process)  │
   │ (in-process) │  │ :5001    │  │ :5002   │  │               │
   │              │  │          │  │         │  │ MossFormer2   │
   │ lazy load,   │  │ conda:   │  │ conda:  │  │ SE + SR       │
   │ unload after │  │ cb_env   │  │ qwen_env│  │               │
   └──────────────┘  └──────────┘  └─────────┘  └───────────────┘
                      ▲ subprocess  ▲ subprocess
                      │ managed by  │ managed by
                      └─ ModelRegistry ─┘
                      ONLY ONE LOADED AT A TIME
```

### 2.2 Directory Structure

```
TTS-Studio/
├── studio.py                       # Main Gradio application
├── requirements.txt                # Base env dependencies
│
├── core/
│   ├── __init__.py
│   ├── transcriber.py              # Faster-Whisper wrapper (lazy load/unload)
│   ├── postprocessing.py           # ClearerVoice wrapper (denoise + super-res)
│   ├── model_registry.py           # Discovers models, manages subprocess + HTTP lifecycle
│   ├── text_processing.py          # Ported from existing: chunking, normalization, pause tags
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
│   └── speaker_name.txt            # Transcript (for Qwen3-TTS, generated via Transcribe button)
│
└── output/                         # All generated audio
```

**Compared to original plan:** Removed `launcher.py` (subprocess management lives in `model_registry.py`), removed `voice_library.py` (voices are just files in a folder), removed `env.yaml` files (conda setup documented in Section 7), added `text_processing.py` (ported from existing code), removed per-voice `.json` metadata files.

---

## 3. Component Specifications

### 3.1 Model Server Protocol (FastAPI)

Each model server exposes a uniform REST API. The main Gradio app is the only client. Servers are started as subprocesses by the Model Registry using `conda run`.

```
POST /generate
  Body (JSON): {
    "text": "Hello world",
    "voice_path": "D:/AI/Balauru's-TTS-Studio/voices/speaker.wav",
    "voice_transcript": "transcript text",     # sent for all models; ignored by models that don't need it
    "parameters": { "temperature": 0.8, ... }  # model-specific, matches config.json parameter names
  }
  Response: WAV file (binary, Content-Type: audio/wav)
  Notes:
    - Each server internally caches voice embeddings keyed by voice_path.
      If the same voice_path is sent again, the server reuses the cached
      embedding instead of recomputing it. This is critical for Qwen3-TTS
      where create_voice_clone_prompt() is expensive.
    - For chunked generation, the main app sends one /generate per chunk.
      The server handles each as a stateless request (but with cached voice).

POST /load
  Loads the TTS model into VRAM. Returns 200 when ready.
  This is called once when switching to this model.

POST /unload
  Frees VRAM (deletes model, empties CUDA cache). Returns 200 when done.

GET /health
  Returns: {"status": "ready" | "loading" | "unloaded", "model_name": "..."}

GET /config
  Returns the model's config.json (capabilities, parameter schema).
  Called once at startup to build the dynamic UI.
```

### 3.2 Model Config Schema (config.json)

Each model declares its capabilities and parameter UI. The Gradio app reads this to dynamically build the interface — no hardcoded model-specific UI.

**Chatterbox config.json:**

```json
{
  "name": "Chatterbox Turbo",
  "id": "chatterbox",
  "port": 5001,
  "conda_env": "cb_env",
  "sample_rate": 24000,
  "needs_transcript": false,
  "needs_text_preprocessing": true,
  "supports_event_tags": true,
  "supported_languages": ["en"],
  "text_preprocessing": {
    "normalize_numbers": true,
    "expand_abbreviations": true,
    "clean_whitespace": true,
    "move_punctuation": true,
    "replace_dashes": true,
    "pause_tags": true
  },
  "parameters": [
    {"name": "temperature",        "type": "slider", "min": 0.05, "max": 2.0, "default": 0.8, "step": 0.05},
    {"name": "top_p",              "type": "slider", "min": 0.0,  "max": 1.0, "default": 0.95, "step": 0.01},
    {"name": "top_k",              "type": "slider", "min": 0,    "max": 1000, "default": 1000, "step": 1},
    {"name": "repetition_penalty", "type": "slider", "min": 1.0,  "max": 2.0, "default": 1.2, "step": 0.01},
    {"name": "min_p",              "type": "slider", "min": 0.0,  "max": 1.0, "default": 0.0, "step": 0.01},
    {"name": "seed",               "type": "number", "default": 0, "label": "Seed (0=Random)"},
    {"name": "norm_loudness",      "type": "checkbox", "default": true, "label": "Normalize Loudness"}
  ],
  "chunk_strategies": [
    "Sentence Batching (<300 chars)",
    "Sentence Batching (<500 chars)",
    "Sentence Split",
    "Paragraph Split",
    "No Split"
  ]
}
```

**Qwen3-TTS Base config.json:**

```json
{
  "name": "Qwen3-TTS Base (Voice Clone)",
  "id": "qwen_tts_base",
  "port": 5002,
  "conda_env": "qwen_env",
  "sample_rate": 24000,
  "needs_transcript": true,
  "needs_text_preprocessing": false,
  "supports_event_tags": false,
  "supported_languages": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
  "text_preprocessing": {},
  "parameters": [
    {"name": "x_vector_only_mode", "type": "checkbox", "default": false,
     "label": "X-Vector only (no transcript needed, lower quality)"},
    {"name": "language",           "type": "dropdown",
     "choices": ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
     "default": "Auto"},
    {"name": "temperature",        "type": "slider", "min": 0.1, "max": 2.0, "default": 0.9, "step": 0.05},
    {"name": "top_k",              "type": "slider", "min": 1,   "max": 200, "default": 50,  "step": 1},
    {"name": "top_p",              "type": "slider", "min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01},
    {"name": "repetition_penalty", "type": "slider", "min": 1.0, "max": 2.0, "default": 1.05, "step": 0.01},
    {"name": "seed",               "type": "number", "default": 0, "label": "Seed (0=Random)"}
  ],
  "chunk_strategies": [
    "Sentence Split",
    "Sentence Batching (<300 chars)",
    "Sentence Batching (<500 chars)",
    "No Split"
  ]
}
```

**Compared to original plan:** Added `conda_env` field (needed for subprocess startup). Fixed Qwen3-TTS parameters to match actual model defaults (`top_k=50`, `temperature=0.9`, `repetition_penalty=1.05`). Added `temperature`, `top_k`, `top_p`, `repetition_penalty` sliders to Qwen3-TTS (they were missing). Added chunking strategies to Qwen3-TTS. Added `norm_loudness` to Chatterbox. Removed `voice_input` field (redundant with `needs_transcript`).

### 3.3 Voice Management

Voices are `.wav` files in the `voices/` folder. Optionally paired with a `.txt` transcript (required for Qwen3-TTS in ICL mode, ignored by Chatterbox).

**No metadata files, no tags, no search.** The UI shows a flat list of voices, lets you pick one, preview it, and transcribe it.

**UI flow:**
1. User drops a `.wav` file into the voice selector (or picks from the dropdown list)
2. If the active model needs a transcript (`needs_transcript: true`), a transcript textbox appears
3. If a `.txt` file exists for the voice, it's loaded automatically
4. If not, user clicks "Transcribe" to generate one via Faster-Whisper
5. User can edit the transcript manually before generating

**File operations:**
- `list_voices()` — scans `voices/` for `.wav` files, returns sorted list
- `get_transcript(voice_name)` — reads `voices/{name}.txt` if it exists, returns `None` otherwise
- `save_transcript(voice_name, text)` — writes `voices/{name}.txt`

These are simple utility functions in `studio.py`, not a separate module.

### 3.4 Transcriber (core/transcriber.py)

Wraps Faster-Whisper for on-demand auto-transcription of voice samples.

```python
class Transcriber:
    def __init__(self, model_size="large-v3"):
        self.model = None  # lazy loaded

    def transcribe(self, audio_path: str) -> str:
        """Load model if needed, transcribe, return text.
        Uses INT8 quantization on GPU if available, falls back to CPU.
        """

    def unload(self):
        """Free model from memory. Called after transcription to
        free VRAM before TTS model loads."""
```

**Behavior:**
- Loads model only when `transcribe()` is called for the first time
- Unloads immediately after transcription completes (VRAM is needed for TTS)
- Default model: `large-v3` (best accuracy for voice sample transcription)

### 3.5 Post-Processing (core/postprocessing.py)

Wraps ClearerVoice-Studio for neural audio enhancement. Runs in the base environment (in-process, not a server). All operations are optional and toggled per-generation.

**Available operations (pipeline order):**

| Step | Task | ClearerVoice Model | What it does |
|------|------|-------------------|--------------|
| 1. Denoise | `speech_enhancement` | `MossFormer2_SE_48K` | Remove background noise, hiss, TTS artifacts |
| 2. Super-Resolution | `speech_super_resolution` | `MossFormer2_SR_48K` | Upsample to 48kHz, restore high frequencies |

**Implementation using actual ClearerVoice API:**

```python
from clearvoice import ClearVoice
import numpy as np
import soundfile as sf

class PostProcessor:
    def __init__(self):
        self._denoise_model = None
        self._sr_model = None

    def process(self, audio_path: str, denoise: bool = False,
                super_resolution: bool = False) -> str:
        """Applies selected pipeline steps in order.
        Returns path to processed file (or original if nothing enabled).
        """
        if denoise:
            if self._denoise_model is None:
                self._denoise_model = ClearVoice(
                    task='speech_enhancement',
                    model_names=['MossFormer2_SE_48K']
                )
            output = self._denoise_model(input_path=audio_path, online_write=False)
            self._denoise_model.write(output, output_path=audio_path)

        if super_resolution:
            if self._sr_model is None:
                self._sr_model = ClearVoice(
                    task='speech_super_resolution',
                    model_names=['MossFormer2_SR_48K']
                )
            output = self._sr_model(input_path=audio_path, online_write=False)
            self._sr_model.write(output, output_path=audio_path)

        return audio_path

    def unload(self):
        """Free ClearerVoice models from VRAM."""
        self._denoise_model = None
        self._sr_model = None
        # torch.cuda.empty_cache() + gc.collect()
```

**Notes:**
- ClearerVoice models auto-download from HuggingFace on first use
- Models are kept loaded between calls for efficiency (they're small: ~200MB each)
- Unload explicitly if VRAM is needed for a large TTS model
- `MossFormer2_SE_48K` expects and outputs 48kHz audio; it internally resamples if input is 24kHz
- `MossFormer2_SR_48K` takes ≥16kHz input and outputs 48kHz

### 3.6 Model Registry (core/model_registry.py)

Discovers available models, manages server subprocesses, enforces single-model-loaded policy. This is the central orchestrator — it replaces the separate `launcher.py` from the original plan.

```python
class ModelRegistry:
    def __init__(self, models_dir="models/"):
        self.models = {}             # id → config dict
        self.active_model = None     # currently loaded model id
        self._processes = {}         # id → subprocess.Popen

    def discover(self):
        """Scan models/ for config.json files, populate self.models."""

    def start_server(self, model_id: str):
        """Start the model's FastAPI server as a subprocess.
        Uses: conda run --no-banner -n {conda_env} python {server.py} --port {port}
        Polls GET /health until status != 'unloaded' or timeout.
        """

    def stop_server(self, model_id: str):
        """Terminate the subprocess. Called when switching models or shutting down."""

    def activate(self, model_id: str):
        """Full model switch sequence:
        1. If another model is active:
           a. POST /unload to current model
           b. stop_server(current_model)
        2. start_server(new_model)
        3. POST /load to new model
        4. Poll GET /health until status == 'ready'
        5. Update self.active_model
        """

    def generate(self, text: str, voice_path: str,
                 voice_transcript: str, params: dict) -> bytes:
        """POST /generate to the active model server.
        Returns raw WAV bytes.
        Raises if no model is active.
        """

    def get_config(self, model_id: str) -> dict:
        """Returns the model's config.json for dynamic UI building."""

    def shutdown(self):
        """Stop all servers. Called on app exit."""
```

**Subprocess management on Windows:**
- Servers are started with `conda run --no-banner -n {conda_env} python models/{id}/server.py --port {port}`
- Each server runs in its own process with its own conda environment
- `stop_server()` sends SIGTERM, waits 5 seconds, then SIGKILL if needed
- On Windows, uses `process.terminate()` then `process.kill()` as fallback

**Health polling:**
- After starting a server, polls `GET /health` every 500ms
- Timeout after 120 seconds (models may need to download on first run)
- If server process exits unexpectedly, surface error to UI

---

## 4. Model Server Implementations

### 4.1 Chatterbox Server (models/chatterbox/server.py)

Wraps the existing `ChatterboxTurboTTS` class. Port existing inference logic from `D:\AI\chatterbox\my_gradio_tts_turbo.py`.

```python
# Key implementation details:
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = None  # loaded on POST /load

@app.post("/load")
def load():
    global model
    model = ChatterboxTurboTTS.from_pretrained("cuda")

@app.post("/generate")
def generate(request: GenerateRequest):
    wav = model.generate(
        request.text,
        audio_prompt_path=request.voice_path,
        temperature=request.parameters.get("temperature", 0.8),
        top_p=request.parameters.get("top_p", 0.95),
        top_k=int(request.parameters.get("top_k", 1000)),
        repetition_penalty=request.parameters.get("repetition_penalty", 1.2),
        min_p=request.parameters.get("min_p", 0.0),
        norm_loudness=request.parameters.get("norm_loudness", True),
    )
    audio = wav.squeeze(0).cpu().numpy()
    # Write to buffer as WAV, return bytes
```

**Ported from existing code:**
- `model.generate()` call with all parameters
- Seed handling (`torch.manual_seed`, etc.)
- Memory cleanup (`torch.cuda.empty_cache()`, `gc.collect()`)

### 4.2 Qwen3-TTS Server (models/qwen_tts/server.py)

Wraps the `Qwen3TTSModel` + `Qwen3TTSTokenizer` classes. The server needs both objects loaded.

```python
# Key implementation details:
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

model = None
tokenizer = None
_voice_cache = {}  # voice_path → VoiceClonePromptItem (reusable speaker embeddings)

@app.post("/load")
def load():
    global model, tokenizer
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # if available
    )
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
    )

@app.post("/generate")
def generate(request: GenerateRequest):
    voice_path = request.voice_path
    x_vector_only = request.parameters.get("x_vector_only_mode", False)

    # Cache voice clone prompt for efficiency (expensive to compute)
    cache_key = f"{voice_path}:{x_vector_only}"
    if cache_key not in _voice_cache:
        _voice_cache[cache_key] = model.create_voice_clone_prompt(
            ref_audio=voice_path,
            ref_text=request.voice_transcript if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )

    wavs, sr = model.generate_voice_clone(
        text=request.text,
        language=request.parameters.get("language", None),  # None = auto
        voice_clone_prompt=_voice_cache[cache_key],
        temperature=request.parameters.get("temperature", 0.9),
        top_k=int(request.parameters.get("top_k", 50)),
        top_p=request.parameters.get("top_p", 1.0),
        repetition_penalty=request.parameters.get("repetition_penalty", 1.05),
    )
    audio = wavs[0]  # np.ndarray, float32, 24kHz
    # Write to buffer as WAV, return bytes
```

**Critical detail — voice clone prompt caching:**
`create_voice_clone_prompt()` extracts speaker embeddings and encodes the reference audio into discrete codes. This is expensive (~2-5 seconds). By caching the result keyed on `voice_path + x_vector_only_mode`, subsequent chunk generations reuse the same prompt instantly. The cache is cleared on `/unload`.

**Language mapping:**
The `language` parameter from the UI dropdown ("Auto", "Chinese", "English", etc.) needs to be mapped to Qwen3-TTS's expected format. "Auto" maps to `None` (model auto-detects).

---

## 5. Text Processing (core/text_processing.py)

Ported directly from the existing Chatterbox Gradio app. Only used when a model's config has `needs_text_preprocessing: true`.

**Functions to port from `my_gradio_tts_turbo.py`:**

| Function | Purpose | Lines in original |
|----------|---------|-------------------|
| `number_to_words(n)` | Convert integers to spoken English | 71-151 |
| `normalize_numbers(text)` | Replace all `\b\d+\b` with words | 153-155 |
| `expand_abbreviations(text)` | "mr" → "mister", "dr" → "doctor", etc. | 157-168 |
| `clean_text_fn(text, ...)` | Whitespace, punctuation, dashes | 170-188 |
| `split_text_into_chunks(text, strategy)` | All chunking strategies | 190-213 |
| `parse_pause_tags(text)` | Parse `[pause:Xs]` into text + silence segments | 215-229 |

**Chunking is shared across all models** (sentence batching, sentence split, paragraph split, no split). Text preprocessing toggles (normalize numbers, expand abbreviations, etc.) are only shown for models that declare them in `text_preprocessing`.

For models with `needs_text_preprocessing: false` (like Qwen3-TTS), only the chunking strategy dropdown appears in the UI. The text is split into chunks but not otherwise modified.

---

## 6. Gradio UI Layout (studio.py)

### 6.1 Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Balauru's TTS Studio                                            │
│  Model: [Chatterbox Turbo ▼]  Status: ● Ready                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Left Column ─────────────────┐  ┌─ Right Column ──────────┐ │
│  │                               │  │                          │ │
│  │  Voice Selector               │  │  Audio Preview           │ │
│  │  [dropdown of voices/ ▼] [▶]  │  │  [waveform player]      │ │
│  │  Transcript: [editable text]  │  │                          │ │
│  │  [Transcribe] (Faster-Whisper)│  │  ── Output Format ──    │ │
│  │                               │  │  (○) wav (○) mp3 (○) flac│ │
│  │  ── Text Input ──             │  │  Speed: [1.0 slider]    │ │
│  │  [multiline textbox]          │  │                          │ │
│  │                               │  │  ── Post-Processing ──  │ │
│  │  ── Event Tags ──             │  │  ☐ Denoise (MossFormer2)│ │
│  │  (only if supports_event_tags)│  │  ☐ Super-Res (→ 48kHz)  │ │
│  │  [laugh] [sigh] [cough] ...   │  │                          │ │
│  │                               │  │  ── History ──          │ │
│  │  ── .txt File Upload ──       │  │  [file selector ▼]      │ │
│  │  [Upload] ☐ Separate files    │  │  [Download ZIP]         │ │
│  │                               │  │                          │ │
│  │  ── Text Processing ──        │  │  ── Parameters ──       │ │
│  │  (only if needs_text_preproc) │  │  (built dynamically     │ │
│  │  Chunking: [strategy ▼]      │  │   from config.json)     │ │
│  │  ☐ Normalize numbers          │  │  Temperature: [slider]  │ │
│  │  ☐ Expand abbreviations       │  │  Top P: [slider]        │ │
│  │  ☐ Clean whitespace           │  │  Top K: [slider]        │ │
│  │  ☐ Move punctuation           │  │  ... etc.               │ │
│  │  ☐ Replace dashes             │  │                          │ │
│  │  ☐ Enable [pause:Xs] tags     │  │                          │ │
│  │                               │  │                          │ │
│  │  [Generate]                   │  │                          │ │
│  │  Status: ...                  │  │                          │ │
│  └───────────────────────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Dynamic UI Behavior (driven by config.json)

| Config field | UI effect |
|---|---|
| `needs_text_preprocessing: true` | Show text preprocessing toggles (whitespace, punctuation, numbers, etc.) |
| `needs_text_preprocessing: false` | Hide text preprocessing toggles; only show chunking strategy dropdown |
| `supports_event_tags: true` | Show event tag buttons below text input |
| `supports_event_tags: false` | Hide event tag section entirely |
| `needs_transcript: true` | Show transcript textbox in voice selector; warn if empty |
| `needs_transcript: false` | Hide transcript textbox |
| `parameters[]` | Dynamically build sliders/dropdowns/checkboxes in Parameters accordion |
| `chunk_strategies[]` | Populate the chunking strategy dropdown |
| `supported_languages[]` | If >1 language, add a language dropdown (auto-populated from config) |

### 6.3 Model Switching

The model dropdown at the top triggers `model_registry.activate(model_id)`. While switching:
1. UI shows "Switching to {model_name}..." status
2. All generation controls are disabled
3. Once `/health` returns `ready`, UI rebuilds dynamic sections from new config
4. Controls re-enable

### 6.4 Generation Flow

```
User clicks [Generate]
  │
  ├─ Read text input (or loaded .txt files)
  ├─ If model needs_text_preprocessing:
  │    Apply selected text cleaning options
  ├─ Split text using selected chunk_strategy
  ├─ If pause_tags enabled: parse [pause:Xs] segments
  │
  ├─ For each chunk:
  │    POST /generate to model server
  │    Receive WAV bytes → numpy array
  │    (Insert silence for pause segments)
  │
  ├─ Concatenate all chunks into final audio
  │
  ├─ If post-processing enabled:
  │    Run ClearerVoice denoise and/or super-resolution
  │
  ├─ If speed != 1.0: apply librosa.effects.time_stretch
  ├─ Convert to output format (wav/mp3/flac)
  ├─ Save to output/ with timestamp filename
  ├─ Update audio preview + history dropdown
  └─ Add to session ZIP
```

---

## 7. VRAM Management

Only one heavy model occupies VRAM at any time. Typical session sequence:

```
1. User opens studio
   → No TTS model loaded. Servers not started.
   → Transcriber not loaded. ClearerVoice not loaded.

2. User clicks "Transcribe" on a voice sample
   → Faster-Whisper loads (~1-2GB VRAM, or CPU if preferred)
   → Transcribes audio → saves .txt next to .wav
   → Faster-Whisper unloads immediately

3. User selects "Chatterbox Turbo" and clicks Generate
   → Model Registry starts chatterbox server subprocess
   → Server loads model into VRAM (~6GB)
   → Generation runs, returns audio
   → Model stays loaded for subsequent generations

4. User enables post-processing on the result
   → ClearerVoice loads MossFormer2 (~200MB each)
   → Processes audio in-place
   → Models stay loaded (small footprint)

5. User switches to "Qwen3-TTS Base"
   → POST /unload to Chatterbox → frees ~6GB VRAM
   → Stop chatterbox server subprocess
   → Start qwen_tts server subprocess
   → POST /load → Qwen3-TTS loads (~4GB for 1.7B with bfloat16)
   → ClearerVoice models remain loaded (still fits)

6. Memory cleanup between chunks (for long texts)
   → Every N chunks: torch.cuda.empty_cache() + gc.collect()
   → Configurable interval (ported from existing code)
```

---

## 8. Environment Setup

### 8.1 Base Environment (UI + transcription + post-processing)

```bash
conda create -n tts_studio python=3.10 -y
conda activate tts_studio
pip install gradio requests
pip install librosa soundfile numpy pydub
pip install faster-whisper
pip install clearvoice
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note:** The base env needs PyTorch with CUDA for ClearerVoice GPU acceleration. Faster-Whisper also benefits from GPU but works on CPU.

### 8.2 Chatterbox Environment

```bash
conda create -n cb_env python=3.10 -y
conda activate cb_env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install chatterbox-tts
pip install fastapi uvicorn
```

**Source reference:** Existing installation at `D:\AI\chatterbox\` uses the same dependencies. The `pyproject.toml` there pins `torch==2.6.0`, `torchaudio==2.6.0`, `transformers==4.46.3`.

### 8.3 Qwen3-TTS Environment

```bash
conda create -n qwen_env python=3.12 -y
conda activate qwen_env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install qwen-tts
pip install fastapi uvicorn
```

**Source reference:** Existing installation at `D:\AI\Qwen3-TTS-Extended\`. The `pyproject.toml` there pins `transformers==4.57.3`, `accelerate==1.12.0`. Flash Attention wheel is available at `D:\AI\Qwen3-TTS-Extended\flash_attn-2.8.3+cu130torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl`.

**Why separate environments:** Chatterbox requires `transformers==4.46.3`, Qwen3-TTS requires `transformers==4.57.3`. These are incompatible.

---

## 9. Implementation Order

### Phase 1: Foundation — Skeleton + Voice Management
1. Create directory structure (`core/`, `models/chatterbox/`, `models/qwen_tts/`, `voices/`, `output/`)
2. Implement `core/audio_utils.py` — concat arrays, insert silence, normalize, format conversion
3. Implement `core/transcriber.py` — Faster-Whisper wrapper with lazy load/unload
4. Basic `studio.py` with voice selector dropdown + transcript textbox + Transcribe button
5. Test: select a voice → click Transcribe → see transcript appear → edit and save

### Phase 2: First Model (Chatterbox Turbo)
6. Write `models/chatterbox/config.json`
7. Write `models/chatterbox/server.py` — FastAPI wrapping `ChatterboxTurboTTS` (port inference from `my_gradio_tts_turbo.py`)
8. Implement `core/model_registry.py` — discover, start_server, stop_server, activate, generate, shutdown
9. Implement `core/text_processing.py` — port chunking + text cleaning from existing code
10. Build dynamic generation UI in `studio.py` (read config.json, build parameter controls, text processing toggles, event tags)
11. Wire up Generate button → text processing → chunked /generate calls → concatenate → play
12. End-to-end test: pick voice → type text → generate → hear audio

### Phase 3: Second Model (Qwen3-TTS Base)
13. Write `models/qwen_tts/config.json`
14. Write `models/qwen_tts/server.py` — FastAPI wrapping `Qwen3TTSModel` + `Qwen3TTSTokenizer` with voice prompt caching
15. Test model switching: Chatterbox → Qwen3-TTS → Chatterbox (verify VRAM is freed correctly)
16. Test transcript-dependent generation: voice + transcript → generate with ICL mode
17. Test x_vector_only_mode: voice without transcript → generate (lower quality but works)

### Phase 4: Post-Processing
18. Implement `core/postprocessing.py` — ClearerVoice wrapper with lazy init
19. Add post-processing checkboxes to UI (denoise + super-resolution)
20. Test pipeline: generate → denoise → super-resolution → play
21. Test VRAM coexistence: TTS model loaded + ClearerVoice running simultaneously

### Phase 5: Polish
22. Output management: generation history dropdown, file selector, session ZIP download
23. `.txt` file upload for batch generation (port from existing code)
24. Output format selector (wav/mp3/flac) + playback speed slider
25. Error handling: server crash recovery, timeout handling, user-facing error messages
26. Status indicators: model loading progress, generation progress (chunk X/Y), VRAM usage

---

## 10. Key Dependencies Summary

| Component | Package | Installed In | GPU |
|---|---|---|---|
| Main UI | `gradio` | tts_studio env | No |
| Model communication | `fastapi`, `uvicorn` | cb_env, qwen_env | No |
| HTTP client | `requests` | tts_studio env | No |
| Audio I/O | `librosa`, `soundfile`, `numpy`, `pydub` | tts_studio env | No |
| Transcription | `faster-whisper` | tts_studio env | Optional (CPU OK) |
| Post-processing | `clearvoice` | tts_studio env | Yes |
| Chatterbox TTS | `chatterbox-tts` | cb_env | Yes |
| Qwen3-TTS | `qwen-tts` | qwen_env | Yes |
| GPU compute | `torch`, `torchaudio` | all three envs | Yes |

---

## 11. Adding Future Models

To add a new TTS model (e.g., F5-TTS, Kokoro, VibeVoice):

1. Create a conda environment with the model's dependencies
2. Create `models/{model_id}/server.py` implementing the 5-endpoint API (`/generate`, `/load`, `/unload`, `/health`, `/config`)
3. Create `models/{model_id}/config.json` declaring the model's name, conda env, port, capabilities, and parameters
4. The Gradio UI automatically discovers and supports it — no UI code changes needed

**The config.json drives everything:** parameter controls, text processing toggles, event tag visibility, transcript requirements, and chunking options are all determined by the config file.
