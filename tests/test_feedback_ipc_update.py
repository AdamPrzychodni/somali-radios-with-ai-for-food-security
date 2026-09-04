"""Tests for weekly impact aggregation and IPC phase adjustment."""

import pandas as pd

from somali_foodsec_radio.feedback.ipc_update import (
    adjust_ipc_phases_with_threshold,
    aggregate_weekly_impact,
)

WEEK = pd.Timestamp("2024-07-22")


def _weekly_row(area="Mudug", week=WEEK, **counts):
    """Build one weekly-impact row with all signal counts defaulting to 0."""
    base = {
        "matched_area": area,
        "week_start": week,
        "drought_warnings": 0,
        "flood_risks": 0,
        "aid_requests": 0,
        "livestock_diseases": 0,
        "rainfall_positives": 0,
        "high_impact_events": 0,
    }
    base.update(counts)
    return base


class TestAggregateWeeklyImpact:
    def test_aggregates_same_area_and_week(self):
        feedback = pd.DataFrame(
            {
                "date": ["2024-07-22", "2024-07-23", "2024-08-05"],
                "matched_area": ["Mudug", "Mudug", "Bay"],
                "drought_warning": [1, 1, 0],
                "flood_risk": [0, 0, 1],
                "aid_request": [0, 0, 0],
                "livestock_disease": [0, 0, 0],
                "rainfall_positive": [0, 0, 0],
                "impact_level": ["high", "high", "high"],
            }
        )
        weekly = aggregate_weekly_impact(feedback)
        mudug = weekly[weekly["matched_area"] == "Mudug"]
        assert len(mudug) == 1
        assert mudug.iloc[0]["drought_warnings"] == 2
        assert mudug.iloc[0]["high_impact_events"] == 2
        assert len(weekly) == 2  # Mudug's week + Bay's week

    def test_drops_unparseable_dates(self):
        feedback = pd.DataFrame(
            {
                "date": ["not a date"],
                "matched_area": ["Mudug"],
                "drought_warning": [1],
                "flood_risk": [0],
                "aid_request": [0],
                "livestock_disease": [0],
                "rainfall_positive": [0],
                "impact_level": ["high"],
            }
        )
        assert aggregate_weekly_impact(feedback).empty


class TestAdjustIpcPhasesWithThreshold:
    def test_raises_phase_when_threshold_met(self):
        geo = pd.DataFrame({"group_name": ["Mudug", "Bay"], "overall_phase_C": [3, 2]})
        weekly = pd.DataFrame([_weekly_row(area="Mudug", drought_warnings=5)])
        result = adjust_ipc_phases_with_threshold(geo, weekly, WEEK)
        phases = result.set_index("group_name")["overall_phase_C"]
        assert phases["Mudug"] == 4  # 3 + 1
        assert phases["Bay"] == 2  # untouched

    def test_sub_threshold_count_changes_nothing(self):
        geo = pd.DataFrame({"group_name": ["Mudug"], "overall_phase_C": [3]})
        weekly = pd.DataFrame([_weekly_row(drought_warnings=4)])  # threshold is 5
        result = adjust_ipc_phases_with_threshold(geo, weekly, WEEK)
        assert result.loc[0, "overall_phase_C"] == 3

    def test_phase_is_clipped_to_five(self):
        geo = pd.DataFrame({"group_name": ["Mudug"], "overall_phase_C": [5]})
        weekly = pd.DataFrame([_weekly_row(drought_warnings=5)])
        result = adjust_ipc_phases_with_threshold(geo, weekly, WEEK)
        assert result.loc[0, "overall_phase_C"] == 5

    def test_phase_is_clipped_to_one(self):
        geo = pd.DataFrame({"group_name": ["Mudug"], "overall_phase_C": [1]})
        weekly = pd.DataFrame([_weekly_row(rainfall_positives=3)])  # effect -1
        result = adjust_ipc_phases_with_threshold(geo, weekly, WEEK)
        assert result.loc[0, "overall_phase_C"] == 1

    def test_unknown_area_is_skipped(self):
        geo = pd.DataFrame({"group_name": ["Bay"], "overall_phase_C": [2]})
        weekly = pd.DataFrame([_weekly_row(area="Mudug", drought_warnings=9)])
        result = adjust_ipc_phases_with_threshold(geo, weekly, WEEK)
        assert result.loc[0, "overall_phase_C"] == 2

    def test_thresholds_come_from_config(self):
        """Editing config/config.yaml must change the output — it used to not."""
        geo = pd.DataFrame({"group_name": ["Mudug"], "overall_phase_C": [3]})
        weekly = pd.DataFrame([_weekly_row(drought_warnings=2)])

        assert (
            adjust_ipc_phases_with_threshold(geo, weekly, WEEK).loc[
                0, "overall_phase_C"
            ]
            == 3
        )
        lowered = adjust_ipc_phases_with_threshold(
            geo,
            weekly,
            WEEK,
            thresholds={"drought_warnings": 2},
            phase_effects={"drought_warnings": 1},
        )
        assert lowered.loc[0, "overall_phase_C"] == 4
