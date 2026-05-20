"""Filesystem path helpers.

Everything here is resolved relative to the repository root so the codebase no longer
depends on hardcoded absolute paths (the old notebooks pointed at
``/teamspace/studios/this_studio/...``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT_MARKER = "pyproject.toml"
_ENV_OVERRIDE = "SOMALI_FOODSEC_RADIO_ROOT"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root directory.

    Resolution order:

    1. The ``SOMALI_FOODSEC_RADIO_ROOT`` environment variable, if set.
    2. The nearest ancestor directory containing ``pyproject.toml``.
    3. As a fallback, two levels above this file (``src/somali_foodsec_radio/``).
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / _ROOT_MARKER).is_file():
            return parent
    return here.parents[2]


def get_data_dir(stage: str = "") -> Path:
    """Return ``<root>/data/<stage>`` (or ``<root>/data`` when *stage* is empty).

    Typical stages: ``raw``, ``interim``, ``interim/transcripts``,
    ``interim/translations``, ``processed``, ``external``.
    """
    base = project_root() / "data"
    return base / stage if stage else base
