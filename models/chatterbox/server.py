"""FastAPI server wrapping ChatterboxTurboTTS.

Run via: conda run --no-banner -n cb_env python models/chatterbox/server.py --port 5001
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
status = "unloaded"  # "unloaded" | "loading" | "ready"


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
    return {"status": status, "model_name": "Chatterbox Turbo"}


@app.get("/config")
def config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


@app.post("/load")
def load():
    global model, status
    status = "loading"
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained("cuda")
        status = "ready"
        return {"status": "ready"}
    except Exception as e:
        status = "unloaded"
        return {"status": "error", "detail": str(e)}


@app.post("/unload")
def unload():
    global model, status
    if model is not None:
        del model
        model = None
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

    # Generate
    wav = model.generate(
        request.text,
        audio_prompt_path=request.voice_path,
        temperature=float(params.get("temperature", 0.8)),
        top_p=float(params.get("top_p", 0.95)),
        top_k=int(params.get("top_k", 1000)),
        repetition_penalty=float(params.get("repetition_penalty", 1.2)),
        min_p=float(params.get("min_p", 0.0)),
        norm_loudness=bool(params.get("norm_loudness", True)),
    )

    audio = wav.squeeze(0).cpu().numpy()

    # Write WAV to buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    _free_memory()

    return Response(content=buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
