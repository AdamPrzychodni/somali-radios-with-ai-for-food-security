"""Aggregate weekly feedback impact and adjust IPC food-security phases.

Thresholds and phase effects come from ``config.yaml`` (``feedback:``) — they are
the model's scientific parameters and must be tunable there, not in code.
"""

from __future__ import annotations

import pandas as pd

from ..config import get_setting


def aggregate_weekly_impact(feedback_matched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched feedback into per-area, per-week impact counts.

    Parses the ``date`` column, drops unparseable rows, derives a ``week_start``, then
    sums the signal columns and counts impact levels per ``(matched_area, week_start)``.
    """
    df = feedback_matched.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    return (
        df.groupby(["matched_area", "week_start"])
        .agg(
            drought_warnings=("drought_warning", "sum"),
            flood_risks=("flood_risk", "sum"),
            aid_requests=("aid_request", "sum"),
            livestock_diseases=("livestock_disease", "sum"),
            rainfall_positives=("rainfall_positive", "sum"),
            high_impact_events=("impact_level", lambda x: (x == "high").sum()),
            medium_impact_events=("impact_level", lambda x: (x == "medium").sum()),
            low_impact_events=("impact_level", lambda x: (x == "low").sum()),
        )
        .reset_index()
    )


def adjust_ipc_phases_with_threshold(
    geo_df,
    weekly_impact_df: pd.DataFrame,
    week: pd.Timestamp,
    thresholds: dict[str, int] | None = None,
    phase_effects: dict[str, int] | None = None,
):
    """Adjust IPC phases for *week* using per-signal thresholds and weighted effects.

    A signal contributes its *phase_effect* only once its weekly count reaches the
    *threshold*. Phases are clipped to 1-5.

    **This is a per-week deviation from the baseline, not a trajectory.** Each call
    starts from the *geo_df* it is given, so passing the same baseline for every week
    yields "what this week's feedback alone implies", and each week is independently
    interpretable. To accumulate instead, feed the returned frame into the next call.

    Args:
        geo_df: IPC GeoDataFrame with ``group_name`` and ``overall_phase_C`` columns.
        weekly_impact_df: Output of :func:`aggregate_weekly_impact`.
        week: The ``week_start`` to score.
        thresholds: Signal -> weekly count needed to shift a phase. Defaults to
            ``feedback.thresholds`` in the config.
        phase_effects: Signal -> phase delta once the threshold is crossed. Defaults
            to ``feedback.phase_effects``.

    Returns:
        A modified copy of *geo_df*.
    """
    if thresholds is None:
        thresholds = get_setting("feedback.thresholds", {})
    if phase_effects is None:
        phase_effects = get_setting("feedback.phase_effects", {})

    updated_geo_df = geo_df.copy()
    week_feedback = weekly_impact_df[weekly_impact_df["week_start"] == week]

    for _, row in week_feedback.iterrows():
        area = row["matched_area"]
        matches = updated_geo_df["group_name"] == area
        if not matches.any():
            continue

        phase_change = sum(
            phase_effects[event_type]
            for event_type, threshold in thresholds.items()
            if row[event_type] >= threshold
        )

        updated_geo_df.loc[matches, "overall_phase_C"] = (
            updated_geo_df.loc[matches, "overall_phase_C"] + phase_change
        ).clip(lower=1, upper=5)

    return updated_geo_df
