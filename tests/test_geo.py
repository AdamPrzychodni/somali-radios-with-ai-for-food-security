"""Tests for geo location matching and transcript loading."""

import pandas as pd

from somali_foodsec_radio.geo.loaders import load_transcripts
from somali_foodsec_radio.geo.matching import (
    assign_geography,
    fuzzy_match_locations,
    match_location_to_geo_df,
    normalize_location_name,
)

# These functions only read DataFrame columns, so a plain DataFrame stands in
# for a GeoDataFrame here.
GEO_DF = pd.DataFrame({"area": ["Mudug", "Hiraan", "Bay", "Gedo"]})
IPC_GEO_DF = pd.DataFrame(
    {"group_name": ["Mudug", "Bay"], "area": ["Galkayo", "Baidoa"]}
)


class TestAssignGeography:
    def test_fuzzy_matches_close_name(self):
        assert assign_geography(["Hiran"], GEO_DF) == ["Hiraan"]

    def test_exact_name_matches(self):
        assert assign_geography(["Gedo"], GEO_DF) == ["Gedo"]

    def test_returns_every_matching_area(self):
        """A bulletin covering three regions is about three regions."""
        assert assign_geography(["Gedo", "Hiran", "Bay"], GEO_DF) == [
            "Gedo",
            "Hiraan",
            "Bay",
        ]

    def test_duplicate_locations_are_not_repeated(self):
        assert assign_geography(["Gedo", "Gedo"], GEO_DF) == ["Gedo"]

    def test_returns_empty_when_nothing_clears_cutoff(self):
        assert assign_geography(["Atlantis"], GEO_DF) == []

    def test_empty_locations_returns_empty(self):
        assert assign_geography([], GEO_DF) == []


class TestNormalizeLocationName:
    def test_drops_region_keyword(self):
        assert normalize_location_name("Mudug Region") == "Mudug"

    def test_drops_urban_keyword(self):
        assert normalize_location_name("Hiraan urban") == "Hiraan"

    def test_strips_digits_and_punctuation(self):
        assert normalize_location_name("Bay District 12!") == "Bay"

    def test_non_string_passes_through(self):
        assert normalize_location_name(123) == 123


class TestMatchLocationToGeoDf:
    def test_exact_match_found(self):
        feedback = pd.DataFrame({"location_normalized": ["Mudug", "Nowhere"]})
        result = match_location_to_geo_df(feedback, IPC_GEO_DF)
        assert result.loc[0, "matched_area"] == "Mudug"
        assert pd.isna(result.loc[1, "matched_area"])  # unmatched -> null

    def test_matches_against_area_column_too(self):
        feedback = pd.DataFrame({"location_normalized": ["Baidoa"]})
        result = match_location_to_geo_df(feedback, IPC_GEO_DF)
        assert result.loc[0, "matched_area"] == "Baidoa"


class TestFuzzyMatchLocations:
    def test_fills_unmatched_with_fuzzy_match(self):
        feedback = pd.DataFrame(
            {"location_normalized": ["Mudugg"], "matched_area": [None]}
        )
        result = fuzzy_match_locations(feedback, IPC_GEO_DF, score_cutoff=70)
        assert result.loc[0, "matched_area"] == "Mudug"

    def test_leaves_none_when_no_good_match(self):
        feedback = pd.DataFrame(
            {"location_normalized": ["Qwxyz"], "matched_area": [None]}
        )
        result = fuzzy_match_locations(feedback, IPC_GEO_DF, score_cutoff=80)
        assert pd.isna(result.loc[0, "matched_area"])


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
