"""FastAPI server wrapping Qwen3-TTS Base (Voice Clone).

Run via: conda run --no-banner -n qwen_env python models/qwen_tts/server.py --port 5002
"""

import argparse
import gc
import io
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

# ── State ──
model = None
tokenizer = None
status = "unloaded"  # "unloaded" | "loading" | "ready"
_voice_cache = {}    # "voice_path:x_vector_only" → VoiceClonePromptItem

# Language display name → Qwen3-TTS language string (case-insensitive matching)
LANGUAGE_MAP = {
    "auto": None,
    "chinese": "chinese",
    "english": "english",
    "japanese": "japanese",
    "korean": "korean",
    "german": "german",
    "french": "french",
    "russian": "russian",
    "portuguese": "portuguese",
    "spanish": "spanish",
    "italian": "italian",
}


class GenerateRequest(BaseModel):
    text: str
    voice_path: str
    voice_transcript: str = ""
    parameters: dict = {}


def _free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@app.get("/health")
def health():
    return {"status": status, "model_name": "Qwen3-TTS Base (Voice Clone)"}


@app.get("/config")
def config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


@app.post("/load")
def load():
    global model, tokenizer, status
    status = "loading"
    try:
        from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map="cuda:0",
        )
        status = "ready"
        return {"status": "ready"}
    except Exception as e:
        status = "unloaded"
        return {"status": "error", "detail": str(e)}


@app.post("/unload")
def unload():
    global model, tokenizer, status, _voice_cache
    _voice_cache.clear()
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    _free_memory()
    status = "unloaded"
    return {"status": "unloaded"}


@app.post("/generate")
def generate(request: GenerateRequest):
    if model is None:
        return Response(content="Model not loaded", status_code=503)

    params = request.parameters

    # Seed
    seed = int(params.get("seed", 0))
    if seed == 0:
        seed = random.randint(1, 1_000_000)
    _set_seed(seed)

    voice_path = request.voice_path
    x_vector_only = bool(params.get("x_vector_only_mode", False))

    # Cache voice clone prompt (expensive to compute: ~2-5s)
    cache_key = f"{voice_path}:{x_vector_only}"
    if cache_key not in _voice_cache:
        _voice_cache[cache_key] = model.create_voice_clone_prompt(
            ref_audio=voice_path,
            ref_text=request.voice_transcript if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )

    # Map language
    lang_str = str(params.get("language", "Auto")).lower()
    language = LANGUAGE_MAP.get(lang_str)

    wavs, sr = model.generate_voice_clone(
        text=request.text,
        language=language,
        voice_clone_prompt=_voice_cache[cache_key],
        temperature=float(params.get("temperature", 0.9)),
        top_k=int(params.get("top_k", 50)),
        top_p=float(params.get("top_p", 1.0)),
        repetition_penalty=float(params.get("repetition_penalty", 1.05)),
    )

    audio = wavs[0]  # np.ndarray, float32, 24kHz

    # Write WAV to buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    _free_memory()

    return Response(content=buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
