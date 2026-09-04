"""Stamp outputs with what produced them.

Transcript and translation CSVs are overwritten in place. Without a record of which
model, which config and which package version made a given row, no evaluation set can
be traced back to the run that produced it — so every output carries these three plus a
timestamp, and a ``.run.json`` sidecar repeats them for the run as a whole.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .config import config_hash

PROVENANCE_COLUMNS = ("model_id", "config_hash", "package_version", "run_timestamp")


def _package_version() -> str:
    try:
        return version("somali-foodsec-radio")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "unknown"


def run_metadata(model_id: str | None = None, **extra: object) -> dict:
    """Return the provenance fields describing the current run."""
    return {
        "model_id": model_id,
        "config_hash": config_hash(),
        "package_version": _package_version(),
        "run_timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def write_run_json(output_path: str | Path, metadata: dict) -> Path:
    """Write *metadata* next to *output_path* as ``<output_path>.run.json``."""
    sidecar = Path(f"{output_path}.run.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return sidecar


def save_with_provenance(
    df: pd.DataFrame,
    output_path: str | Path,
    model_id: str | None = None,
    **extra: object,
) -> Path:
    """Write *df* to CSV with provenance columns, plus a ``.run.json`` sidecar."""
    metadata = run_metadata(model_id, **extra)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(**metadata).to_csv(output_path, index=False)
    write_run_json(output_path, metadata)
    return output_path
