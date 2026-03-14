# SPDX-License-Identifier: Apache-2.0
"""
Text-to-Speech (TTS) engine using mlx-audio.

Supports:
- Kokoro (fast, lightweight)
- Chatterbox (multilingual, expressive)
- VibeVoice (realtime, low latency)
- VoxCPM (Chinese/English, high quality)
"""

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default models
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

# Available voices per model family
KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

CHATTERBOX_VOICES = ["default"]  # Uses reference audio for voice


@dataclass
class AudioOutput:
    """Output from TTS generation."""

    audio: np.ndarray
    sample_rate: int
    duration: float


class TTSEngine:
    """
    Text-to-Speech engine supporting multiple model families.

    Usage:
        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine.load()
        audio = engine.generate("Hello world!", voice="af_heart")
        engine.save(audio, "output.wav")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
    ):
        """
        Initialize TTS engine.

        Args:
            model_name: HuggingFace model name. Supported families:
                - Kokoro: mlx-community/Kokoro-82M-bf16, Kokoro-82M-4bit
                - Chatterbox: mlx-community/chatterbox-turbo-fp16
                - VibeVoice: mlx-community/VibeVoice-Realtime-0.5B-4bit
                - VoxCPM: mlx-community/VoxCPM1.5
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._model_family = self._detect_family(model_name)

    def _detect_family(self, model_name: str) -> str:
        """Detect model family from name."""
        name_lower = model_name.lower()
        if "kokoro" in name_lower:
            return "kokoro"
        elif "chatterbox" in name_lower:
            return "chatterbox"
        elif "vibevoice" in name_lower:
            return "vibevoice"
        elif "voxcpm" in name_lower:
            return "voxcpm"
        elif "csm" in name_lower:
            return "csm"
        elif "cosyvoice" in name_lower:
            return "cosyvoice"
        else:
            return "kokoro"  # Default

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        try:
            from mlx_audio.tts.generate import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(
                f"TTS model loaded: {self.model_name} (family: {self._model_family})"
            )
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> AudioOutput:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (model-specific)
            speed: Speech speed (0.5 to 2.0)
            lang_code: Language code (a=English, e=Spanish, f=French, etc.)

        Returns:
            AudioOutput with audio data and metadata
        """
        if not self._loaded:
            self.load()

        try:
            import mlx.core as mx

            audio_chunks = []
            sample_rate = 24000  # Default for most models

            for result in self.model.generate(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
            ):
                audio_data = result.audio
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

                # Convert mlx array to numpy
                if isinstance(audio_data, mx.array) or hasattr(audio_data, "tolist"):
                    audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio_data, dtype=np.float32)

                audio_chunks.append(audio_np)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            full_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )
            duration = len(full_audio) / sample_rate

            return AudioOutput(
                audio=full_audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def stream_generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Iterator[AudioOutput]:
        """
        Stream speech generation chunk by chunk.

        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed

        Yields:
            AudioOutput chunks
        """
        if not self._loaded:
            self.load()

        sample_rate = 24000

        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
        ):
            audio_data = result.audio
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

            if hasattr(audio_data, "tolist"):
                audio_np = np.array(audio_data.tolist(), dtype=np.float32)
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            yield AudioOutput(
                audio=audio_np,
                sample_rate=sample_rate,
                duration=len(audio_np) / sample_rate,
            )

    def save(
        self,
        audio: AudioOutput,
        path: str | Path,
        format: str = "wav",
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: AudioOutput to save
            path: Output file path
            format: Output format (wav, mp3)
        """
        try:
            from mlx_audio.tts import save_audio

            save_audio(audio.audio, str(path), sample_rate=audio.sample_rate)
            logger.info(f"Audio saved to {path}")
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav

            # Ensure audio is in correct format
            audio_int16 = (audio.audio * 32767).astype(np.int16)
            wav.write(str(path), audio.sample_rate, audio_int16)
            logger.info(f"Audio saved to {path} (scipy fallback)")

    def to_bytes(
        self,
        audio: AudioOutput,
        format: str = "wav",
    ) -> bytes:
        """
        Convert audio to bytes.

        Args:
            audio: AudioOutput to convert
            format: Output format (wav, mp3)

        Returns:
            Audio data as bytes
        """
        import scipy.io.wavfile as wav

        buffer = io.BytesIO()
        audio_int16 = (audio.audio * 32767).astype(np.int16)
        wav.write(buffer, audio.sample_rate, audio_int16)
        return buffer.getvalue()

    def get_voices(self) -> list:
        """Get available voices for current model."""
        if self._model_family == "kokoro":
            return KOKORO_VOICES
        elif self._model_family == "chatterbox":
            return CHATTERBOX_VOICES
        else:
            return ["default"]

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("TTS model unloaded")


def generate_speech(
    text: str,
    model_name: str = DEFAULT_TTS_MODEL,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> AudioOutput:
    """
    Convenience function to generate speech without managing engine.

    Args:
        text: Text to synthesize
        model_name: Model to use
        voice: Voice ID
        speed: Speech speed

    Returns:
        AudioOutput
    """
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, voice=voice, speed=speed)
