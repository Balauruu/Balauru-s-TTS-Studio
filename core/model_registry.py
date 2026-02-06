"""Model Registry — discovers models, manages server subprocesses,
enforces single-model-loaded policy.

Each model lives in models/{id}/ with a config.json and server.py.
Servers run as subprocesses via ``conda run`` in isolated environments.
"""

import json
import os
import subprocess
import sys
import time

import requests


class ModelRegistry:
    def __init__(self, models_dir: str | None = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.models_dir = models_dir
        self.models: dict[str, dict] = {}        # id → config dict
        self.active_model: str | None = None      # currently loaded model id
        self._processes: dict[str, subprocess.Popen] = {}  # id → subprocess

    # ── Discovery ──────────────────────────────────────────────

    def discover(self):
        """Scan models/ for subdirectories containing config.json."""
        self.models.clear()
        if not os.path.isdir(self.models_dir):
            return
        for entry in os.listdir(self.models_dir):
            config_path = os.path.join(self.models_dir, entry, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                model_id = config.get("id", entry)
                config["_dir"] = os.path.join(self.models_dir, entry)
                self.models[model_id] = config

    # ── Server lifecycle ───────────────────────────────────────

    def start_server(self, model_id: str):
        """Start the model's FastAPI server as a conda subprocess."""
        config = self.models[model_id]
        server_py = os.path.join(config["_dir"], "server.py")
        conda_env = config["conda_env"]
        port = config["port"]

        cmd = [
            "conda", "run", "--no-banner", "-n", conda_env,
            "python", server_py, "--port", str(port),
        ]

        # On Windows, use CREATE_NEW_PROCESS_GROUP for clean termination
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        self._processes[model_id] = proc

        # Poll /health until the server is up
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 120  # 120s timeout
        while time.time() < deadline:
            # Check if process died
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise RuntimeError(
                    f"Server for {model_id} exited with code {proc.returncode}: {stderr}"
                )
            try:
                resp = requests.get(f"{base_url}/health", timeout=2)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)

        # Timeout — kill the process
        self.stop_server(model_id)
        raise TimeoutError(f"Server for {model_id} did not start within 120 seconds")

    def stop_server(self, model_id: str):
        """Terminate the server subprocess."""
        proc = self._processes.pop(model_id, None)
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    # ── Model switching ────────────────────────────────────────

    def activate(self, model_id: str):
        """Full model switch: unload current → start new → load new."""
        if model_id not in self.models:
            raise ValueError(f"Unknown model: {model_id}")

        # If same model already active, skip
        if self.active_model == model_id:
            return

        # Deactivate current model
        if self.active_model is not None:
            self._deactivate_current()

        # Start new server
        self.start_server(model_id)

        # POST /load to load the model into VRAM
        port = self.models[model_id]["port"]
        base_url = f"http://127.0.0.1:{port}"
        resp = requests.post(f"{base_url}/load", timeout=300)
        resp.raise_for_status()

        # Poll until ready
        deadline = time.time() + 300
        while time.time() < deadline:
            resp = requests.get(f"{base_url}/health", timeout=5)
            data = resp.json()
            if data.get("status") == "ready":
                self.active_model = model_id
                return
            time.sleep(0.5)

        raise TimeoutError(f"Model {model_id} did not become ready within 300 seconds")

    def _deactivate_current(self):
        """Unload the current model and stop its server."""
        model_id = self.active_model
        if model_id is None:
            return
        port = self.models[model_id]["port"]
        base_url = f"http://127.0.0.1:{port}"
        try:
            requests.post(f"{base_url}/unload", timeout=30)
        except Exception:
            pass  # Server might already be dead
        self.stop_server(model_id)
        self.active_model = None

    # ── Generation ─────────────────────────────────────────────

    def generate(self, text: str, voice_path: str,
                 voice_transcript: str, params: dict) -> bytes:
        """POST /generate to the active model. Returns raw WAV bytes."""
        if self.active_model is None:
            raise RuntimeError("No model is active")

        port = self.models[self.active_model]["port"]
        base_url = f"http://127.0.0.1:{port}"

        payload = {
            "text": text,
            "voice_path": voice_path,
            "voice_transcript": voice_transcript,
            "parameters": params,
        }

        resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.content

    # ── Config access ──────────────────────────────────────────

    def get_config(self, model_id: str) -> dict:
        """Return a model's config dict."""
        return self.models[model_id]

    # ── Shutdown ───────────────────────────────────────────────

    def shutdown(self):
        """Stop all servers. Called on app exit."""
        if self.active_model:
            self._deactivate_current()
        for model_id in list(self._processes.keys()):
            self.stop_server(model_id)
