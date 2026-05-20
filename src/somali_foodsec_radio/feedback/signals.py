"""Detect food-security impact signals in Radio Ergo caller feedback."""

from __future__ import annotations

import pandas as pd

# Signal name -> keywords searched (whole-word) in each call's remarks. The signal
# names are fixed: infer_impact_level and the weekly aggregation depend on them.
DEFAULT_IMPACT_SIGNALS: dict[str, list[str]] = {
    "drought_warning": ["drought", "water shortage", "dry"],
    "flood_risk": ["flood", "river", "overflow"],
    "aid_request": ["aid", "help", "assistance", "displacement", "idp"],
    "livestock_disease": ["livestock", "goats", "sick", "disease", "cattle", "death"],
    "rainfall_positive": ["rain", "rainfall", "good rain"],
}


def create_impact_signals(
    feedback_df: pd.DataFrame,
    signal_keywords: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add one 0/1 column per impact signal, based on keywords found in ``remarks``.

    Args:
        feedback_df: Feedback DataFrame with a ``remarks`` column.
        signal_keywords: Signal name -> keyword list. Defaults to
            :data:`DEFAULT_IMPACT_SIGNALS`.

    Returns:
        A copy of *feedback_df* with one integer column per signal.
    """
    signal_keywords = signal_keywords or DEFAULT_IMPACT_SIGNALS
    df = feedback_df.copy()
    remarks_lower = df["remarks"].str.lower().fillna("")

    for signal, keywords in signal_keywords.items():
        pattern = "|".join(rf"\b{keyword}\b" for keyword in keywords)
        df[signal] = remarks_lower.str.contains(pattern, regex=True).astype(int)

    return df


def infer_impact_level(row: pd.Series) -> str:
    """Infer an impact level (``high`` / ``medium`` / ``low`` / ``unknown``) for a row.

    Uses the theme text and the 0/1 signal columns added by
    :func:`create_impact_signals`.
    """
    theme = str(row["theme"]).lower()

    if row["drought_warning"] == 1 or "drought" in theme:
        return "high"
    if row["flood_risk"] == 1 or "flood" in theme:
        return "high"
    if "conflict" in theme:
        return "high"
    if row["livestock_disease"] == 1 or "livestock" in theme:
        return "medium"
    if row["rainfall_positive"] == 1 or "rain" in theme:
        return "low"
    if row["aid_request"] == 1 or "aid" in theme:
        return "low"
    if "environment" in theme:
        return "medium"

    return "unknown"
