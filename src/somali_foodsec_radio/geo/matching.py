"""Match place names to IPC geographic areas via fuzzy string matching."""

from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from ..text_utils import clean_text


def assign_geography(
    locations: list[str],
    geo_df,
    area_col: str = "area",
    score_cutoff: int = 80,
) -> str:
    """Fuzzy-match extracted *locations* to an IPC area name.

    Args:
        locations: Extracted place names.
        geo_df: GeoDataFrame (or DataFrame) with an *area_col* column.
        area_col: Column holding IPC area names.
        score_cutoff: Minimum similarity score (0-100).

    Returns:
        The best-matching area name, or ``"Unknown"`` if nothing clears the cutoff.
    """
    area_list = geo_df[area_col].dropna().tolist()
    clean_areas = [clean_text(a) for a in area_list]
    best = ("Unknown", 0)
    for loc in locations:
        loc_clean = clean_text(loc)
        match, score, _ = process.extractOne(
            query=loc_clean,
            choices=clean_areas,
            scorer=fuzz.token_sort_ratio,
        )
        if match and score >= score_cutoff and score > best[1]:
            best = (area_list[clean_areas.index(match)], score)
    return best[0]


def normalize_location_name(name: str) -> str:
    """Normalise a raw location name for matching.

    Lowercases, drops the words ``region``/``urban``/``district``, strips non-letter
    characters, collapses whitespace, and title-cases the result. Non-string input is
    returned unchanged.
    """
    if not isinstance(name, str):
        return name
    name = name.lower()
    name = re.sub(r"\b(region|urban|district)\b", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def match_location_to_geo_df(feedback_df: pd.DataFrame, geo_df) -> pd.DataFrame:
    """Exact-match ``location_normalized`` values to geo ``group_name``/``area`` names.

    Returns a copy of *feedback_df* with a ``matched_area`` column (``None`` where no
    exact match was found).
    """
    df = feedback_df.copy()
    geo_locations = set(geo_df["group_name"].dropna().str.lower()) | set(
        geo_df["area"].dropna().str.lower()
    )
    df["matched_area"] = [
        loc.title() if loc.lower() in geo_locations else None
        for loc in df["location_normalized"]
    ]
    return df


def fuzzy_match_locations(
    feedback_df: pd.DataFrame, geo_df, score_cutoff: int = 80
) -> pd.DataFrame:
    """Fuzzy-match feedback rows still missing a ``matched_area``.

    Returns a copy of *feedback_df* with ``matched_area`` filled in where a fuzzy
    match clears *score_cutoff*.
    """
    df = feedback_df.copy()
    geo_areas = (
        pd.concat(
            [
                geo_df["group_name"].dropna().str.lower(),
                geo_df["area"].dropna().str.lower(),
            ]
        )
        .unique()
        .tolist()
    )
    for idx, row in df[df["matched_area"].isna()].iterrows():
        loc = row["location_normalized"].lower()
        result = process.extractOne(loc, geo_areas)
        # rapidfuzz returns (choice, score, index), or None when there are no choices.
        if result is not None and result[1] >= score_cutoff:
            df.at[idx, "matched_area"] = result[0].title()
        else:
            df.at[idx, "matched_area"] = None
    return df
