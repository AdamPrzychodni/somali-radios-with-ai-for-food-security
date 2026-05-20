"""Tests for the YAML configuration loader."""

from pathlib import Path

import pytest

from somali_foodsec_radio.config import _deep_merge, load_config


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_reads_yaml(tmp_path):
    cfg = _write(
        tmp_path / "config.yaml",
        "project:\n  name: demo\nasr:\n  batch_size: 8\n",
    )
    config = load_config(cfg)
    assert config["project"]["name"] == "demo"
    assert config["asr"]["batch_size"] == 8


def test_load_config_resolves_paths_to_absolute(tmp_path):
    cfg = _write(tmp_path / "config.yaml", "paths:\n  data_raw: data/raw\n")
    resolved = load_config(cfg)["paths"]["data_raw"]
    assert resolved.is_absolute()
    assert resolved.name == "raw"


def test_local_override_is_deep_merged(tmp_path):
    _write(tmp_path / "config.yaml", "asr:\n  batch_size: 8\n  use_fp16: true\n")
    _write(tmp_path / "config.local.yaml", "asr:\n  batch_size: 4\n")
    config = load_config(tmp_path / "config.yaml")
    assert config["asr"]["batch_size"] == 4       # overridden
    assert config["asr"]["use_fp16"] is True      # sibling key preserved


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 99}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 99}}
    assert base == {"a": {"x": 1, "y": 2}}        # base untouched


def test_repo_config_yaml_loads():
    """The real config/config.yaml at the repo root must load and look sane."""
    config = load_config()
    assert config["project"]["name"] == "somali_foodsec_radio"
    assert config["paths"]["data_raw"].is_absolute()
    assert config["asr"]["wav2vec2_model"] == "Mustafaa4a/ASR-Somali"
