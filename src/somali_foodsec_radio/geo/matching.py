"""Match place names to IPC geographic areas via fuzzy string matching."""

from __future__ import annotations

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
