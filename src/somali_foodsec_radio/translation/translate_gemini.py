"""Gemini text translation."""

from __future__ import annotations

import os


def translate_text(
    text: str,
    source_language: str = "Somali",
    target_language: str = "English",
    model: str = "gemini-2.0-flash",
) -> str | None:
    """Translate *text* between languages with the Gemini API.

    Args:
        text: The text to translate.
        source_language: The source language name.
        target_language: The target language name.
        model: The Gemini model to use.

    Returns:
        The translated text.

    Raises:
        EnvironmentError: If ``GEMINI_API_KEY`` is not set.
        ValueError: If the API response contains no text.
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise OSError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)
    prompt = (
        f"Translate the following {source_language} text to "
        f"{target_language}:\n\n{text}"
    )

    try:
        response = client.models.generate_content(model=model, contents=[prompt])
    except Exception as exc:  # noqa: BLE001 - wrap any API error
        raise RuntimeError(f"Failed to generate translation: {exc}") from exc

    if hasattr(response, "text"):
        return response.text
    raise ValueError("No translation text found in the response.")
