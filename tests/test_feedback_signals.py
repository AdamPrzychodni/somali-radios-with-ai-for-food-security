"""Tests for impact-signal detection and impact-level inference."""

import pandas as pd
import pytest

from somali_foodsec_radio.feedback.signals import (
    create_impact_signals,
    infer_impact_level,
    signal_present,
)


def _row(theme="", **signals):
    """Build a feedback row with all signal columns defaulting to 0."""
    base = {
        "theme": theme,
        "drought_warning": 0,
        "flood_risk": 0,
        "aid_request": 0,
        "livestock_disease": 0,
        "rainfall_positive": 0,
    }
    base.update(signals)
    return pd.Series(base)


class TestCreateImpactSignals:
    def test_detects_signals_from_remarks(self):
        df = pd.DataFrame(
            {
                "remarks": [
                    "A severe drought and water shortage",
                    "Good rain this season",
                    "Just a normal update",
                ]
            }
        )
        result = create_impact_signals(df)
        assert result["drought_warning"].tolist() == [1, 0, 0]
        assert result["rainfall_positive"].tolist() == [0, 1, 0]

    def test_drops_helper_columns(self):
        result = create_impact_signals(pd.DataFrame({"remarks": ["x"]}))
        assert "remarks_lower" not in result.columns
        assert "theme_lower" not in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"remarks": ["drought"]})
        create_impact_signals(df)
        assert list(df.columns) == ["remarks"]

    def test_missing_remarks_are_not_a_signal(self):
        df = pd.DataFrame({"remarks": [None]})
        result = create_impact_signals(df)
        assert result["drought_warning"].tolist() == [0]


class TestNegation:
    """A drought report must not register as a food-security improvement.

    `rainfall_positive` is worth -1 IPC phase, so a bare `\brain\b` match on
    "no rain" reported an improvement on the worst possible input.
    """

    @pytest.mark.parametrize(
        "remark",
        [
            "there has been no rain for months",
            "the rains failed again this season",
            "we are still waiting for rain",
            "rain is needed urgently",
            "not a drop of rain since June",
        ],
    )
    def test_negated_rainfall_is_not_positive(self, remark):
        df = create_impact_signals(pd.DataFrame({"remarks": [remark]}))
        assert df["rainfall_positive"].iloc[0] == 0, remark

    @pytest.mark.parametrize(
        "remark",
        [
            "good rain this season",
            "heavy rain filled the berkads",
            "rainfall has improved the pasture",
        ],
    )
    def test_genuine_rainfall_still_counts(self, remark):
        df = create_impact_signals(pd.DataFrame({"remarks": [remark]}))
        assert df["rainfall_positive"].iloc[0] == 1, remark

    def test_one_negated_mention_does_not_cancel_a_genuine_one(self):
        remark = "no rain in April but good rain arrived in May"
        df = create_impact_signals(pd.DataFrame({"remarks": [remark]}))
        assert df["rainfall_positive"].iloc[0] == 1

    def test_dry_season_is_not_a_drought_warning(self):
        df = create_impact_signals(pd.DataFrame({"remarks": ["the dry season began"]}))
        assert df["drought_warning"].iloc[0] == 0

    def test_human_casualties_are_not_livestock_disease(self):
        remark = "three people died in the flooding"
        df = create_impact_signals(pd.DataFrame({"remarks": [remark]}))
        assert df["livestock_disease"].iloc[0] == 0

    def test_multi_word_keyword_matches(self):
        df = create_impact_signals(pd.DataFrame({"remarks": ["a water shortage here"]}))
        assert df["drought_warning"].iloc[0] == 1

    def test_negator_outside_the_window_does_not_cancel(self):
        tokens = "no one expected it but the good rain came"
        assert signal_present(tokens, ["rain"], ["no"], window=3) is True


class TestInferImpactLevel:
    def test_drought_signal_is_high(self):
        assert infer_impact_level(_row(drought_warning=1)) == "high"

    def test_flood_signal_is_high(self):
        assert infer_impact_level(_row(flood_risk=1)) == "high"

    def test_conflict_theme_is_high(self):
        assert infer_impact_level(_row(theme="Conflict in the region")) == "high"

    def test_livestock_signal_is_medium(self):
        assert infer_impact_level(_row(livestock_disease=1)) == "medium"

    def test_rainfall_signal_is_low(self):
        assert infer_impact_level(_row(rainfall_positive=1)) == "low"

    def test_aid_signal_is_low(self):
        assert infer_impact_level(_row(aid_request=1)) == "low"

    def test_environment_theme_is_medium(self):
        assert infer_impact_level(_row(theme="environment update")) == "medium"

    def test_nothing_matches_is_unknown(self):
        assert infer_impact_level(_row(theme="general news")) == "unknown"
