"""Tests for impact-signal detection and impact-level inference."""

import pandas as pd

from somali_foodsec_radio.feedback.signals import (
    create_impact_signals,
    infer_impact_level,
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
