"""Tests for geo location matching and transcript loading."""

import pandas as pd

from somali_foodsec_radio.geo.loaders import load_transcripts
from somali_foodsec_radio.geo.matching import assign_geography

# assign_geography only reads an `area` column, so a plain DataFrame stands in
# for a GeoDataFrame here.
GEO_DF = pd.DataFrame({"area": ["Mudug", "Hiraan", "Bay", "Gedo"]})


class TestAssignGeography:
    def test_fuzzy_matches_close_name(self):
        assert assign_geography(["Hiran"], GEO_DF) == "Hiraan"

    def test_exact_name_matches(self):
        assert assign_geography(["Gedo"], GEO_DF) == "Gedo"

    def test_returns_unknown_when_nothing_clears_cutoff(self):
        assert assign_geography(["Atlantis"], GEO_DF) == "Unknown"

    def test_empty_locations_returns_unknown(self):
        assert assign_geography([], GEO_DF) == "Unknown"


class TestLoadTranscripts:
    def test_parses_date_from_filename(self, tmp_path):
        (tmp_path / "idaacadda-01-jul-2024.txt").write_text("first", encoding="utf-8")
        (tmp_path / "idaacadda-02-jul-2024.txt").write_text("second", encoding="utf-8")
        df = load_transcripts(str(tmp_path))
        assert list(df.columns) == ["file", "date", "text"]
        assert len(df) == 2
        assert df.iloc[0]["date"] == pd.Timestamp("2024-07-01")
        assert df.iloc[0]["text"] == "first"

    def test_missing_date_becomes_nat(self, tmp_path):
        (tmp_path / "no-date-here.txt").write_text("x", encoding="utf-8")
        df = load_transcripts(str(tmp_path))
        assert pd.isna(df.iloc[0]["date"])

    def test_empty_directory_returns_empty_frame(self, tmp_path):
        assert load_transcripts(str(tmp_path)).empty
