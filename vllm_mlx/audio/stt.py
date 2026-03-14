# SPDX-License-Identifier: Apache-2.0
"""
Speech-to-Text (STT) engine using mlx-audio.

Supports:
- Whisper (multilingual, 99+ languages)
- Parakeet (English-focused, fast)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default models
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list | None = None


class STTEngine:
    """
    Speech-to-Text engine supporting Whisper and Parakeet models.

    Usage:
        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        engine.load()
        result = engine.transcribe("audio.mp3")
        print(result.text)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
    ):
        """
        Initialize STT engine.

        Args:
            model_name: HuggingFace model name. Supported:
                - mlx-community/whisper-large-v3-mlx (multilingual)
                - mlx-community/whisper-large-v3-turbo (fast)
                - mlx-community/whisper-medium-mlx
                - mlx-community/whisper-small-mlx
                - mlx-community/parakeet-tdt-0.6b-v2 (English, fastest)
                - mlx-community/parakeet-tdt-0.6b-v3
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._is_parakeet = "parakeet" in model_name.lower()

    def load(self) -> None:
        """Load the STT model."""
        if self._loaded:
            return

        try:
            from mlx_audio.stt.utils import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(f"STT model loaded: {self.model_name}")
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for STT. Install with: pip install mlx-audio"
            ) from e

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code (e.g., "en", "es"). Auto-detected if None.
            task: "transcribe" or "translate" (translate to English)

        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        try:
            # Use the model's generate method directly
            kwargs = {"verbose": False}
            if language and not self._is_parakeet:
                kwargs["language"] = language
            if task and not self._is_parakeet:
                kwargs["task"] = task

            result = self.model.generate(audio_path, **kwargs)

            # Extract text and metadata from result
            text = getattr(result, "text", str(result)) if result else ""
            segments = getattr(result, "segments", None)
            detected_lang = getattr(result, "language", None)

            # Calculate duration from segments if available
            duration = None
            if segments:
                last_seg = segments[-1] if segments else None
                if last_seg and hasattr(last_seg, "end"):
                    duration = last_seg.end

            return TranscriptionResult(
                text=text.strip() if isinstance(text, str) else str(text),
                language=detected_lang,
                duration=duration,
                segments=segments,
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("STT model unloaded")


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> TranscriptionResult:
    """
    Convenience function to transcribe audio without managing engine.

    Args:
        audio_path: Path to audio file
        model_name: Model to use
        language: Language code (optional)

    Returns:
        TranscriptionResult
    """
    engine = STTEngine(model_name)
    engine.load()
    return engine.transcribe(audio_path, language=language)
