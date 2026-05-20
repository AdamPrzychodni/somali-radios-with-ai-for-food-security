"""Pair extracted themes with IPC geographies, and the end-to-end topic pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..geo.loaders import load_geojson, load_transcripts
from ..geo.matching import assign_geography
from .bertopic_model import fit_topic_model, select_themes_per_doc
from .locations import extract_locations


def build_theme_location_pairs(
    df: pd.DataFrame,
    topic_ids: list[int],
    themes_list: list[list[str]],
    geo_df,
) -> pd.DataFrame:
    """Create one row per ``(file, date, theme, geography)``.

    Args:
        df: Transcripts DataFrame (needs a ``text`` column).
        topic_ids: Primary topic per document (kept for reference; unused here).
        themes_list: List of themes per document.
        geo_df: IPC boundaries.

    Returns:
        Exploded DataFrame with columns ``['file', 'date', 'theme', 'geography']``.
    """
    df = df.copy()
    df["themes"] = themes_list
    df["locations"] = df["text"].apply(extract_locations)
    df["geography"] = df["locations"].apply(
        lambda locs: assign_geography(locs, geo_df)
    )
    exploded = df.explode("themes")
    return exploded[["file", "date", "themes", "geography"]].rename(
        columns={"themes": "theme"}
    )


def run_topic_pipeline(
    transcripts_dir: str,
    geojson_path: str,
    theme_map: dict[int, str],
    model_kwargs: dict[str, Any] | None = None,
    prob_threshold: float = 0.1,
) -> pd.DataFrame:
    """Run end-to-end: load data, topic-model, multi-theme, pair themes with locations.

    Returns:
        DataFrame with one row per theme and its matched geography.
    """
    geo_df = load_geojson(geojson_path)
    transcripts_df = load_transcripts(transcripts_dir)
    docs = transcripts_df["text"].tolist()

    _, topic_ids, probabilities = fit_topic_model(docs, model_kwargs)
    themes_list = select_themes_per_doc(probabilities, theme_map, prob_threshold)

    return build_theme_location_pairs(
        df=transcripts_df,
        topic_ids=topic_ids,
        themes_list=themes_list,
        geo_df=geo_df,
    )
