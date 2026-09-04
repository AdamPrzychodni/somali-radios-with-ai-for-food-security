"""Plot Somalia IPC phase maps and weekly feedback-signal series."""

from __future__ import annotations

from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from ..config import get_setting

# IPC phase colours (light green -> dark red), matching the official IPC maps.
# Used when `geo.phase_colors` is absent from the config.
PHASE_COLORS = {
    1: "#B7E4C7",
    2: "#FFE066",
    3: "#FF9F1C",
    4: "#FF4040",
    5: "#800020",
}


def phase_colors() -> dict[int, str]:
    """IPC phase -> colour, from ``geo.phase_colors`` in the config."""
    configured = get_setting("geo.phase_colors", PHASE_COLORS)
    return {int(phase): colour for phase, colour in configured.items()}


def plot_ipc_maps(
    geo_df,
    mode: Literal["current", "projected", "both"] = "both",
) -> None:
    """Plot IPC phase maps for the current period, projected period, or both.

    Args:
        geo_df: GeoDataFrame with IPC phase columns.
        mode: ``'current'``, ``'projected'`` or ``'both'``.
    """
    if mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        periods = [
            ("Current", "overall_phase_C", geo_df["current_period_dates"].iloc[0]),
            ("Projected", "overall_phase_P", geo_df["projected_period_dates"].iloc[0]),
        ]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 12))
        periods = []
        if mode == "current":
            periods.append(
                ("Current", "overall_phase_C", geo_df["current_period_dates"].iloc[0])
            )
        elif mode == "projected":
            periods.append(
                (
                    "Projected",
                    "overall_phase_P",
                    geo_df["projected_period_dates"].iloc[0],
                )
            )
        axes = [axes]

    for ax, (title, phase_column, period_dates) in zip(axes, periods, strict=False):
        geo_df["color"] = geo_df[phase_column].map(phase_colors())

        # Plot each phase separately so the legend can be controlled.
        for phase, color in phase_colors().items():
            subset = geo_df[geo_df[phase_column] == phase]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)

        ax.set_title(f"{title} Period\n{period_dates}", fontsize=16)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True)
        ax.axis("equal")

    patches = [
        mpatches.Patch(color=color, label=f"Phase {phase}")
        for phase, color in phase_colors().items()
    ]
    fig.legend(
        handles=patches, title="IPC Phase", loc="lower center", ncol=5, fontsize=12
    )

    plt.suptitle("Somalia IPC Maps", fontsize=22)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def plot_ipc_map_single(geo_df, week) -> None:
    """Plot the IPC ``overall_phase_C`` map for a single week.

    Args:
        geo_df: GeoDataFrame with an ``overall_phase_C`` column.
        week: The week-start timestamp, used in the title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    geo_df["color"] = geo_df["overall_phase_C"].map(phase_colors())

    for phase, color in phase_colors().items():
        subset = geo_df[geo_df["overall_phase_C"] == phase]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)

    ax.set_title(
        f"IPC Phase Map - Week Starting {week.strftime('%Y-%m-%d')}", fontsize=16
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True)
    ax.axis("equal")

    patches = [
        mpatches.Patch(color=color, label=f"Phase {phase}")
        for phase, color in phase_colors().items()
    ]
    fig.legend(
        handles=patches, title="IPC Phase", loc="lower center", ncol=5, fontsize=12
    )

    plt.suptitle("Somalia IPC Map (Dynamic Update)", fontsize=22)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def plot_weekly_signals(weekly_impact_df: pd.DataFrame, area: str) -> None:
    """Plot the weekly feedback-signal time series for a single area."""
    area_data = weekly_impact_df[weekly_impact_df["matched_area"] == area]
    if area_data.empty:
        print(f"No data for {area}")
        return

    area_data = area_data.sort_values("week_start")
    series = {
        "drought_warnings": "Drought Warnings",
        "flood_risks": "Flood Risks",
        "aid_requests": "Aid Requests",
        "livestock_diseases": "Livestock Diseases",
        "rainfall_positives": "Positive Rainfall",
    }

    plt.figure(figsize=(14, 7))
    for column, label in series.items():
        plt.plot(area_data["week_start"], area_data[column], label=label, marker="o")
    plt.title(f"Weekly Feedback Signals in {area}", fontsize=16)
    plt.xlabel("Week Start", fontsize=12)
    plt.ylabel("Number of Reports", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
