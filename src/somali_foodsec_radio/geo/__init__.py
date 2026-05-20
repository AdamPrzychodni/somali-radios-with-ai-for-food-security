"""IPC geometries, location matching and map plotting.

``geopandas`` is imported lazily by :func:`~somali_foodsec_radio.geo.loaders.load_geojson`,
so importing this package is cheap.
"""

from .loaders import load_geojson, load_transcripts
from .matching import assign_geography
from .plotting import PHASE_COLORS, plot_ipc_maps

__all__ = [
    "PHASE_COLORS",
    "assign_geography",
    "load_geojson",
    "load_transcripts",
    "plot_ipc_maps",
]
