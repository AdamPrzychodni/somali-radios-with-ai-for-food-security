"""Load IPC GeoJSON boundaries and transcript text files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_geojson(file_path: str):
    """Load a GeoJSON file into a GeoDataFrame.

    ``geopandas`` is imported lazily, so this module stays importable without the
    ``[analysis]`` extras.

    Returns:
        geopandas.GeoDataFrame

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    import geopandas as gpd

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"GeoJSON file not found: {file_path}")
    try:
        return gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as ValueError below
        raise ValueError(f"Error loading GeoJSON: {exc}") from exc


def load_transcripts(transcripts_dir: str) -> pd.DataFrame:
    """Read ``.txt`` transcripts from a directory, parsing the date from each filename.

    Returns:
        DataFrame with columns ``['file', 'date', 'text']``.
    """
    records: list[dict[str, Any]] = []
    for txt_path in sorted(Path(transcripts_dir).glob("*.txt")):
        match = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", txt_path.name)
        date = (
            pd.to_datetime(match.group(1), format="%d-%b-%Y") if match else pd.NaT
        )
        text = txt_path.read_text(encoding="utf-8")
        records.append({"file": txt_path.name, "date": date, "text": text})
    return pd.DataFrame(records)
