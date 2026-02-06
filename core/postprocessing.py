"""Post-processing via ClearerVoice-Studio.

Provides optional denoise (speech_enhancement) and super-resolution
(speech_super_resolution) using MossFormer2 models. Runs in-process
in the base tts_studio environment.
"""

import gc
import torch


class PostProcessor:
    def __init__(self):
        self._denoise_model = None
        self._sr_model = None

    def process(
        self,
        audio_path: str,
        denoise: bool = False,
        super_resolution: bool = False,
    ) -> str:
        """Apply selected pipeline steps in order.

        Returns the path to the processed file (same path — processed in-place).
        If nothing is enabled, returns the original path unchanged.
        """
        if denoise:
            self._ensure_denoise_model()
            output = self._denoise_model(input_path=audio_path, online_write=False)
            self._denoise_model.write(output, output_path=audio_path)

        if super_resolution:
            self._ensure_sr_model()
            output = self._sr_model(input_path=audio_path, online_write=False)
            self._sr_model.write(output, output_path=audio_path)

        return audio_path

    def _ensure_denoise_model(self):
        if self._denoise_model is None:
            from clearvoice import ClearVoice
            self._denoise_model = ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )

    def _ensure_sr_model(self):
        if self._sr_model is None:
            from clearvoice import ClearVoice
            self._sr_model = ClearVoice(
                task="speech_super_resolution",
                model_names=["MossFormer2_SR_48K"],
            )

    def unload(self):
        """Free ClearerVoice models from VRAM."""
        self._denoise_model = None
        self._sr_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
