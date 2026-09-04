"""IPC geometries, location matching and map plotting.

``geopandas`` is imported lazily by :func:`~somali_foodsec_radio.geo.loaders.load_geojson`,
so importing this package is cheap.
"""

from .loaders import load_geojson, load_transcripts
from .matching import (
    assign_geography,
    fuzzy_match_locations,
    match_location_to_geo_df,
    normalize_location_name,
)
from .plotting import (
    PHASE_COLORS,
    phase_colors,
    plot_ipc_map_single,
    plot_ipc_maps,
    plot_weekly_signals,
)

__all__ = [
    "PHASE_COLORS",
    "assign_geography",
    "fuzzy_match_locations",
    "load_geojson",
    "load_transcripts",
    "match_location_to_geo_df",
    "normalize_location_name",
    "phase_colors",
    "plot_ipc_map_single",
    "plot_ipc_maps",
    "plot_weekly_signals",
]
