"""Location extraction from transcripts via spaCy named-entity recognition."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_setting


@lru_cache(maxsize=1)
def _get_nlp():
    """Load the spaCy model once, on first use.

    Lazy so that importing this module does not require spaCy or download a model.
    """
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "spaCy is required for location extraction. Install it with: "
            "pip install -e '.[analysis]'"
        ) from exc

    model_name = get_setting("topics.spacy_model", "en_core_web_sm")
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - depends on downloaded model
        raise OSError(
            f"spaCy model '{model_name}' is not installed. Run: "
            f"python -m spacy download {model_name}"
        ) from exc


def extract_locations(text: str) -> list[str]:
    """Extract GPE/LOC entity strings from *text* via spaCy NER."""
    doc = _get_nlp()(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
