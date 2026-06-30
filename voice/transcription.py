"""
RxPilot — Voice Transcription Module.

Transcribes audio files (WAV, MP3, WEBM, OGG) to text using OpenAI Whisper.

The module loads the Whisper "base" model (74 MB, CPU-compatible).
If whisper is not installed, a stub transcription is returned so that
tests and CI can run without the heavy dependency.

Usage:
    from voice.transcription import transcribe_audio

    text = transcribe_audio("/path/to/audio.wav")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

# Model name — "base" is 74MB and runs in ~3s on CPU
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Lazy-loaded model
_whisper_model = None
_whisper_available: bool | None = None


def _is_whisper_available() -> bool:
    """Check if openai-whisper is installed."""
    global _whisper_available
    if _whisper_available is None:
        try:
            import whisper  # noqa: F401
            _whisper_available = True
        except ImportError:
            _whisper_available = False
            logger.warning(
                "openai-whisper not installed. "
                "Voice transcription will use stub mode. "
                "Install with: pip install openai-whisper"
            )
    return _whisper_available


def _get_model():
    """Lazy-load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper model: %s", WHISPER_MODEL_NAME)
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        logger.info("Whisper model loaded")
    return _whisper_model


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file to text.

    Args:
        file_path: Absolute path to the audio file.

    Returns:
        Transcribed text string.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not _is_whisper_available():
        # Stub mode — return a placeholder for dev/CI without Whisper
        logger.info("Stub transcription for: %s", path.name)
        return _stub_transcription(path.name)

    try:
        model = _get_model()
        logger.info("Transcribing: %s", path.name)
        result = model.transcribe(str(path), language="en", fp16=False)
        transcript = result.get("text", "").strip()
        logger.info(
            "Transcribed %d characters from %s", len(transcript), path.name
        )
        return transcript
    except Exception as e:
        logger.error("Transcription failed for %s: %s", path.name, e)
        raise RuntimeError(f"Transcription failed: {e}") from e


def _stub_transcription(filename: str) -> str:
    """
    Return a stub transcript when Whisper is not available.

    Useful for CI and local dev without a Whisper installation.
    The stub varies based on filename to allow testing different intent paths.
    """
    name_lower = filename.lower()

    if "stock" in name_lower or "inventory" in name_lower:
        return "What is the current stock level of Metformin?"
    elif "expir" in name_lower:
        return "Which medicines are expiring in the next 30 days?"
    elif "interact" in name_lower or "interact" in name_lower:
        return "Are there any drug interactions between Warfarin and Aspirin?"
    else:
        # Generic pharmacy query
        return "What medicines did we receive in the last delivery?"
