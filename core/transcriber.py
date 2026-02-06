import gc
import torch


class Transcriber:
    """Wraps Faster-Whisper for on-demand transcription of voice samples.

    Lazy-loads the model on first use and unloads immediately after
    transcription to free VRAM for TTS models.
    """

    def __init__(self, model_size: str = "large-v3"):
        self.model_size = model_size
        self.model = None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text.

        Loads the model if not already loaded, transcribes, then
        immediately unloads to free VRAM.
        """
        if self.model is None:
            self._load()

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=None,  # auto-detect
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments)

        # Unload immediately — VRAM is needed for TTS
        self.unload()

        return text

    def _load(self):
        """Load the Faster-Whisper model."""
        from faster_whisper import WhisperModel

        # Use GPU with INT8 quantization if available, else CPU
        if torch.cuda.is_available():
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="int8_float16",
            )
        else:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def unload(self):
        """Free the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
