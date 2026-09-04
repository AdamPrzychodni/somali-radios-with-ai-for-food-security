"""Load and validate project configuration from ``config/config.yaml``.

Configuration is plain YAML. A gitignored ``config/config.local.yaml`` (if present) is
deep-merged on top, so machine-specific overrides never reach version control. The
merged result is validated against :mod:`somali_foodsec_radio.config_schema` before any
path resolution, so a typo fails at load rather than mid-run. Every value under the
``paths:`` section is then resolved to an absolute path against the repository root.
"""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import config_schema
from .paths import project_root

_CONFIG_DIR = "config"
_MAIN_FILE = "config.yaml"
_LOCAL_FILE = "config.local.yaml"
_PATHS_SECTION = "paths"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged key by key; any other value in *override* replaces the
    one in *base*. Neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_paths(config: dict, root: Path) -> dict:
    """Turn every value in the ``paths:`` section into an absolute :class:`Path`."""
    paths = config.get(_PATHS_SECTION)
    if isinstance(paths, dict):
        config[_PATHS_SECTION] = {
            name: (root / str(value)).resolve() for name, value in paths.items()
        }
    return config


def load_config(path: str | Path | None = None, validate: bool = True) -> dict:
    """Load, validate and return the project configuration as a dict.

    Args:
        path: Optional explicit path to the main YAML file. Defaults to
            ``<project_root>/config/config.yaml``.
        validate: Check the merged config against the schema before returning it.

    Raises:
        FileNotFoundError: if the main config file does not exist.
        pydantic.ValidationError: if the merged config does not match the schema.
    """
    root = project_root()
    main_path = Path(path) if path else root / _CONFIG_DIR / _MAIN_FILE
    if not main_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {main_path}. Expected config/config.yaml at the "
            f"repository root."
        )

    with open(main_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    local_path = main_path.with_name(_LOCAL_FILE)
    if local_path.is_file():
        with open(local_path, encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        config = _deep_merge(config, local)

    if validate:
        config_schema.validate(config)

    return _resolve_paths(config, root)


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Return the cached project configuration (loaded once per process)."""
    return load_config()


def get_setting(dotted_key: str, default: Any = None) -> Any:
    """Look up a nested config value by dotted key.

    Example::

        get_setting("asr.batch_size", default=8)

    Returns *default* if any part of the path is missing.
    """
    node: Any = get_config()
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def config_hash(config: dict | None = None) -> str:
    """Short stable digest of the configuration, for stamping onto outputs.

    Paths are excluded: they are machine-specific and would make otherwise identical
    runs look different.
    """
    payload = {k: v for k, v in (config or get_config()).items() if k != _PATHS_SECTION}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
