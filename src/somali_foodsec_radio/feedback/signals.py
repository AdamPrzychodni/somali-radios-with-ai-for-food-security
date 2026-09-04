"""Detect food-security impact signals in Radio Ergo caller feedback.

Keyword lists, negators and the negation window all come from ``config.yaml``
(``feedback:``) — nothing here duplicates them as a constant.

The matching is negation-aware on purpose. A plain word-boundary search for
``rain`` fires on *"no rain"*, *"the rains failed"* and *"still waiting for rain"*,
each of which sets ``rainfall_positive`` — worth **-1 IPC phase**. That reports a
food-security improvement on a drought report.
"""

from __future__ import annotations

import re

import pandas as pd

from ..config import get_setting

_WORD = re.compile(r"[a-z']+")

# Signal names are fixed: infer_impact_level and the weekly aggregation depend on them.
SIGNAL_NAMES = (
    "drought_warning",
    "flood_risk",
    "aid_request",
    "livestock_disease",
    "rainfall_positive",
)


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _keyword_spans(tokens: list[str], keyword: str):
    """Yield ``(start, end)`` token spans where *keyword* occurs in *tokens*."""
    words = keyword.lower().split()
    span = len(words)
    for i in range(len(tokens) - span + 1):
        if tokens[i : i + span] == words:
            yield i, i + span


def signal_present(
    remarks: str,
    keywords: list[str],
    negators: set[str] | list[str],
    window: int = 3,
) -> bool:
    """True if any *keyword* occurs in *remarks* without a negator nearby.

    A negator within *window* tokens either side of the match cancels it, so
    *"no rain for months"* and *"the rains failed"* do not count as rainfall.
    """
    tokens = _tokenize(remarks)
    negators = set(negators)
    for keyword in keywords:
        for start, end in _keyword_spans(tokens, keyword):
            context = (
                tokens[max(0, start - window) : start] + tokens[end : end + window]
            )
            if not negators.intersection(context):
                return True
    return False


def create_impact_signals(
    feedback_df: pd.DataFrame,
    signal_keywords: dict[str, list[str]] | None = None,
    negators: list[str] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Add one 0/1 column per impact signal, from keywords found in ``remarks``.

    Args:
        feedback_df: Feedback DataFrame with a ``remarks`` column.
        signal_keywords: Signal name -> keyword list. Defaults to
            ``feedback.impact_signals`` in the config.
        negators: Words that cancel a nearby keyword match. Defaults to
            ``feedback.negators``.
        window: Tokens either side of a match searched for a negator. Defaults to
            ``feedback.negation_window``.

    Returns:
        A copy of *feedback_df* with one integer column per signal.
    """
    signal_keywords = signal_keywords or get_setting("feedback.impact_signals", {})
    negators = (
        negators if negators is not None else get_setting("feedback.negators", [])
    )
    window = (
        window if window is not None else get_setting("feedback.negation_window", 3)
    )

    df = feedback_df.copy()
    remarks = df["remarks"].fillna("")

    for signal, keywords in signal_keywords.items():
        df[signal] = remarks.map(
            lambda text, kw=keywords: int(signal_present(text, kw, negators, window))
        )

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
