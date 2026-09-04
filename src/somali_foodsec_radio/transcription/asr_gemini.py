"""Gemini audio transcription."""

from __future__ import annotations

import os


def transcribe_audio(file_path: str, model: str = "gemini-2.0-flash") -> str | None:
    """Transcribe an audio file with the Gemini API.

    Args:
        file_path: Path to the audio (.mp3) file.
        model: The Gemini model to use.

    Returns:
        The transcript text.

    Raises:
        EnvironmentError: If ``GEMINI_API_KEY`` is not set.
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the API response contains no text.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise OSError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    try:
        with open(file_path, "rb") as fh:
            audio_bytes = fh.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Audio file not found: {file_path}") from exc
    except Exception as exc:  # noqa: BLE001 - wrap any read error
        raise RuntimeError(f"Failed to read file: {exc}") from exc

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    prompt = "Generate a transcript of the speech."

    try:
        response = client.models.generate_content(
            model=model, contents=[prompt, audio_part]
        )
    except Exception as exc:  # noqa: BLE001 - wrap any API error
        raise RuntimeError(f"Failed to generate content: {exc}") from exc

    if hasattr(response, "text"):
        return response.text
    raise ValueError("No transcript text found in the response.")
