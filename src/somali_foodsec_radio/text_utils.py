"""Shared text-normalisation helpers.

Lives at the package root (rather than inside ``topics``) because both ``topics`` and
``geo`` use :func:`clean_text` — keeping it here avoids a circular import.
"""

from __future__ import annotations

import string
import unicodedata


def clean_text(text: str) -> str:
    """Normalise *text*: lowercase, strip accents and punctuation, trim whitespace."""
    lowered = text.lower()
    normalized = unicodedata.normalize("NFD", lowered)
    no_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    no_punct = no_accents.translate(str.maketrans("", "", string.punctuation))
    return no_punct.strip()
