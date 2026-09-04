"""Pair extracted themes with IPC geographies, and the end-to-end topic pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import get_setting
from ..geo.loaders import load_geojson, load_transcripts
from ..geo.matching import assign_geography
from .bertopic_model import fit_topic_model, select_themes_per_doc, verify_theme_map
from .locations import extract_locations


def build_theme_location_pairs(
    df: pd.DataFrame,
    themes_list: list[list[str]],
    geo_df,
) -> pd.DataFrame:
    """Create one row per ``(file, date, theme, geography)``.

    Both themes and geographies are exploded: a broadcast covering three regions
    produces three rows per theme. Documents that matched no area keep a ``None``
    geography rather than being dropped.

    Args:
        df: Transcripts DataFrame (needs a ``text`` column).
        themes_list: List of themes per document.
        geo_df: IPC boundaries.
    """
    df = df.copy()
    df["themes"] = themes_list
    df["locations"] = df["text"].apply(extract_locations)
    df["geographies"] = df["locations"].apply(
        lambda locs: assign_geography(locs, geo_df) or [None]
    )
    exploded = df.explode("themes").explode("geographies")
    return exploded[["file", "date", "themes", "geographies"]].rename(
        columns={"themes": "theme", "geographies": "geography"}
    )


def run_topic_pipeline(
    transcripts_dir: str,
    geojson_path: str,
    theme_map: dict[int, str],
    model_kwargs: dict[str, Any] | None = None,
    prob_threshold: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run end-to-end: load data, topic-model, multi-theme, pair themes with locations.

    Raises if the fitted topics no longer match *theme_map* — see
    :func:`~somali_foodsec_radio.topics.bertopic_model.verify_theme_map`.
    """
    if prob_threshold is None:
        prob_threshold = get_setting("topics.prob_threshold", 0.1)

    geo_df = load_geojson(geojson_path)
    transcripts_df = load_transcripts(transcripts_dir)
    docs = transcripts_df["text"].tolist()

    model, _, probabilities = fit_topic_model(docs, model_kwargs, seed=seed)
    verify_theme_map(model, theme_map)
    themes_list = select_themes_per_doc(probabilities, theme_map, prob_threshold)

    return build_theme_location_pairs(
        df=transcripts_df,
        themes_list=themes_list,
        geo_df=geo_df,
    )
