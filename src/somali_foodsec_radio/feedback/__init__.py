"""Parse Radio Ergo caller-feedback PDFs and update IPC food-security phases.

Pipeline: :func:`~somali_foodsec_radio.feedback.pdf_extract.extract_detailed_calls_from_pdf`
-> :func:`~somali_foodsec_radio.feedback.signals.create_impact_signals` /
:func:`~somali_foodsec_radio.feedback.signals.infer_impact_level` -> location matching
(see :mod:`somali_foodsec_radio.geo.matching`) ->
:func:`~somali_foodsec_radio.feedback.ipc_update.aggregate_weekly_impact` -> phase
adjustment.
"""

from .ipc_update import (
    PHASE_EFFECTS,
    THRESHOLDS,
    adjust_ipc_phases,
    adjust_ipc_phases_with_threshold,
    aggregate_weekly_impact,
    plot_time_series,
)
from .pdf_extract import clean_date, extract_detailed_calls_from_pdf, is_valid_date
from .signals import DEFAULT_IMPACT_SIGNALS, create_impact_signals, infer_impact_level

__all__ = [
    "DEFAULT_IMPACT_SIGNALS",
    "PHASE_EFFECTS",
    "THRESHOLDS",
    "adjust_ipc_phases",
    "adjust_ipc_phases_with_threshold",
    "aggregate_weekly_impact",
    "clean_date",
    "create_impact_signals",
    "extract_detailed_calls_from_pdf",
    "infer_impact_level",
    "is_valid_date",
    "plot_time_series",
]
