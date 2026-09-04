"""Tests for output provenance stamping."""

import json

import pandas as pd

from somali_foodsec_radio.provenance import (
    PROVENANCE_COLUMNS,
    run_metadata,
    save_with_provenance,
)


def test_run_metadata_carries_every_column():
    meta = run_metadata("some/model")
    assert set(PROVENANCE_COLUMNS) <= set(meta)
    assert meta["model_id"] == "some/model"
    assert meta["package_version"] != "unknown"


def test_saved_csv_records_what_produced_it(tmp_path):
    df = pd.DataFrame({"transcript_text": ["hello", "world"]})
    out = save_with_provenance(df, tmp_path / "sub" / "t.csv", model_id="engine/v1")

    written = pd.read_csv(out)
    assert list(written["model_id"]) == ["engine/v1", "engine/v1"]
    assert set(written["config_hash"]) == {run_metadata()["config_hash"]}

    sidecar = json.loads(out.with_name("t.csv.run.json").read_text())
    assert sidecar["model_id"] == "engine/v1"
    assert sidecar["config_hash"] == written["config_hash"].iloc[0]


def test_extra_fields_reach_the_sidecar(tmp_path):
    out = save_with_provenance(
        pd.DataFrame({"x": [1]}), tmp_path / "t.csv", model_id="m", batch_size=8
    )
    sidecar = json.loads(out.with_name("t.csv.run.json").read_text())
    assert sidecar["batch_size"] == 8


def test_input_frame_is_not_mutated(tmp_path):
    df = pd.DataFrame({"x": [1]})
    save_with_provenance(df, tmp_path / "t.csv")
    assert list(df.columns) == ["x"]
