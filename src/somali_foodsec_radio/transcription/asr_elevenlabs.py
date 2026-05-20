"""ElevenLabs Scribe speech-to-text."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path


def transcribe_with_elevenlabs(
    audio_path: str | Path,
    model_id: str = "scribe_v1",
    api_key: str | None = None,
) -> str:
    """Transcribe an audio file with the ElevenLabs Scribe API.

    Args:
        audio_path: Path to the audio file.
        model_id: ElevenLabs speech-to-text model (only ``scribe_v1`` is supported).
        api_key: API key; falls back to the ``ELEVENLABS_API_KEY`` environment variable.

    Returns:
        The transcript text.

    Raises:
        EnvironmentError: If no API key is provided or found in the environment.
    """
    from elevenlabs.client import ElevenLabs

    api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ELEVENLABS_API_KEY not found. Set it in your .env file."
        )

    client = ElevenLabs(api_key=api_key)
    with open(audio_path, "rb") as fh:
        audio_data = BytesIO(fh.read())

    transcription = client.speech_to_text.convert(file=audio_data, model_id=model_id)
    return transcription.text
