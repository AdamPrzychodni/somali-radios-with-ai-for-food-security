"""Topic modelling and theme extraction from translated transcripts.

``bertopic`` and ``spacy`` are imported lazily by the functions that need them, so
importing this package is cheap.
"""

from ..text_utils import clean_text
from .bertopic_model import fit_topic_model, select_themes_per_doc
from .locations import extract_locations
from .theme_location import build_theme_location_pairs, run_topic_pipeline

__all__ = [
    "build_theme_location_pairs",
    "clean_text",
    "extract_locations",
    "fit_topic_model",
    "run_topic_pipeline",
    "select_themes_per_doc",
]
