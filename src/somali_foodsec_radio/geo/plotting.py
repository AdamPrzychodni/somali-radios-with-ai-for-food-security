"""Plot Somalia IPC phase maps."""

from __future__ import annotations

from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# IPC phase colours (light green -> dark red), matching the official IPC maps.
PHASE_COLORS = {
    1: "#B7E4C7",  # light green
    2: "#FFE066",  # yellow
    3: "#FF9F1C",  # orange
    4: "#FF4040",  # red
    5: "#800020",  # dark red
}


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
        geo_df["color"] = geo_df[phase_column].map(PHASE_COLORS)

        # Plot each phase separately so the legend can be controlled.
        for phase, color in PHASE_COLORS.items():
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
        for phase, color in PHASE_COLORS.items()
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

    geo_df["color"] = geo_df["overall_phase_C"].map(PHASE_COLORS)

    for phase, color in PHASE_COLORS.items():
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
        for phase, color in PHASE_COLORS.items()
    ]
    fig.legend(
        handles=patches, title="IPC Phase", loc="lower center", ncol=5, fontsize=12
    )

    plt.suptitle("Somalia IPC Map (Dynamic Update)", fontsize=22)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
