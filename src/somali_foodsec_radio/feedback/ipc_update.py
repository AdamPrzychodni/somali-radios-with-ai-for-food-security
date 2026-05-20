"""Aggregate weekly feedback impact and adjust IPC food-security phases."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

# Weekly signal count needed to shift an IPC phase.
THRESHOLDS: dict[str, int] = {
    "drought_warnings": 5,
    "flood_risks": 3,
    "aid_requests": 5,
    "livestock_diseases": 3,
    "rainfall_positives": 3,
}

# IPC phase delta applied when a signal crosses its threshold.
PHASE_EFFECTS: dict[str, int] = {
    "drought_warnings": +1,
    "flood_risks": +1,
    "aid_requests": +1,
    "livestock_diseases": +1,
    "rainfall_positives": -1,
}


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


def adjust_ipc_phases(
    geo_df, weekly_impact_df: pd.DataFrame, week: pd.Timestamp
):
    """Adjust IPC phases for *week*: high-impact events raise, rainfall lowers.

    Phases are clipped to the valid 1-5 range. Returns a modified copy of *geo_df*.
    """
    updated_geo_df = geo_df.copy()
    week_feedback = weekly_impact_df[weekly_impact_df["week_start"] == week]

    for _, row in week_feedback.iterrows():
        area = row["matched_area"]
        if area not in updated_geo_df["group_name"].values:
            continue
        idx = updated_geo_df[updated_geo_df["group_name"] == area].index

        phase_change = row["high_impact_events"] - row["rainfall_positives"]
        updated_geo_df.loc[idx, "overall_phase_C"] = (
            updated_geo_df.loc[idx, "overall_phase_C"] + phase_change
        ).clip(lower=1, upper=5)

    return updated_geo_df


def adjust_ipc_phases_with_threshold(
    geo_df,
    weekly_impact_df: pd.DataFrame,
    week: pd.Timestamp,
    thresholds: dict[str, int] | None = None,
    phase_effects: dict[str, int] | None = None,
):
    """Adjust IPC phases for *week* using per-signal thresholds and weighted effects.

    A signal contributes its *phase_effect* only once its weekly count reaches the
    *threshold*. Phases are clipped to 1-5. Returns a modified copy of *geo_df*.
    """
    thresholds = thresholds if thresholds is not None else THRESHOLDS
    phase_effects = phase_effects if phase_effects is not None else PHASE_EFFECTS

    updated_geo_df = geo_df.copy()
    week_feedback = weekly_impact_df[weekly_impact_df["week_start"] == week]

    for _, row in week_feedback.iterrows():
        area = row["matched_area"]
        if area not in updated_geo_df["group_name"].values:
            continue
        idx = updated_geo_df[updated_geo_df["group_name"] == area].index

        phase_change = 0
        for event_type, threshold in thresholds.items():
            if row[event_type] >= threshold:
                phase_change += phase_effects[event_type]

        updated_geo_df.loc[idx, "overall_phase_C"] = (
            updated_geo_df.loc[idx, "overall_phase_C"] + phase_change
        ).clip(lower=1, upper=5)

    return updated_geo_df


def plot_time_series(weekly_impact_df: pd.DataFrame, area: str) -> None:
    """Plot the weekly feedback-signal time series for a single area."""
    area_data = weekly_impact_df[weekly_impact_df["matched_area"] == area]
    if area_data.empty:
        print(f"No data for {area}")
        return

    area_data = area_data.sort_values("week_start")

    plt.figure(figsize=(14, 7))
    plt.plot(
        area_data["week_start"], area_data["drought_warnings"],
        label="Drought Warnings", marker="o",
    )
    plt.plot(
        area_data["week_start"], area_data["flood_risks"],
        label="Flood Risks", marker="o",
    )
    plt.plot(
        area_data["week_start"], area_data["aid_requests"],
        label="Aid Requests", marker="o",
    )
    plt.plot(
        area_data["week_start"], area_data["livestock_diseases"],
        label="Livestock Diseases", marker="o",
    )
    plt.plot(
        area_data["week_start"], area_data["rainfall_positives"],
        label="Positive Rainfall", marker="o",
    )
    plt.title(f"Weekly Feedback Signals in {area}", fontsize=16)
    plt.xlabel("Week Start", fontsize=12)
    plt.ylabel("Number of Reports", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
