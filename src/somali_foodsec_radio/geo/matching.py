"""Match place names to IPC geographic areas via fuzzy string matching."""

from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from ..config import get_setting
from ..text_utils import clean_text


def assign_geography(
    locations: list[str],
    geo_df,
    area_col: str | None = None,
    score_cutoff: int | None = None,
) -> list[str]:
    """Fuzzy-match extracted *locations* to IPC area names.

    Returns **every** distinct area that clears the cutoff, not just the best one:
    a bulletin covering three regions is about three regions. An empty list means
    nothing matched — the same "no match" sentinel the rest of this module uses.

    Args:
        locations: Extracted place names.
        geo_df: GeoDataFrame (or DataFrame) with an *area_col* column.
        area_col: Column holding IPC area names. Defaults to ``geo.area_col``.
        score_cutoff: Minimum similarity score (0-100). Defaults to
            ``geo.fuzzy_score_cutoff``.
    """
    if area_col is None:
        area_col = get_setting("geo.area_col", "area")
    if score_cutoff is None:
        score_cutoff = get_setting("geo.fuzzy_score_cutoff", 80)

    area_list = geo_df[area_col].dropna().tolist()
    clean_areas = [clean_text(a) for a in area_list]
    if not clean_areas:
        return []

    matched: list[str] = []
    for loc in locations:
        match, score, index = process.extractOne(
            query=clean_text(loc),
            choices=clean_areas,
            scorer=fuzz.token_sort_ratio,
        )
        if score >= score_cutoff and area_list[index] not in matched:
            matched.append(area_list[index])
    return matched


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
    feedback_df: pd.DataFrame, geo_df, score_cutoff: int | None = None
) -> pd.DataFrame:
    """Fuzzy-match feedback rows still missing a ``matched_area``.

    Returns a copy of *feedback_df* with ``matched_area`` filled in where a fuzzy
    match clears *score_cutoff* (default: ``geo.fuzzy_score_cutoff``).
    """
    if score_cutoff is None:
        score_cutoff = get_setting("geo.fuzzy_score_cutoff", 80)

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
            df.loc[idx, "matched_area"] = result[0].title()
        else:
            df.loc[idx, "matched_area"] = None
    return df
