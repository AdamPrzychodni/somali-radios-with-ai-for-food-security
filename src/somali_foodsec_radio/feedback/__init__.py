"""Parse Radio Ergo caller-feedback PDFs and update IPC food-security phases.

Pipeline: :func:`~somali_foodsec_radio.feedback.pdf_extract.extract_detailed_calls_from_pdf`
-> :func:`~somali_foodsec_radio.feedback.signals.create_impact_signals` /
:func:`~somali_foodsec_radio.feedback.signals.infer_impact_level` -> location matching
(see :mod:`somali_foodsec_radio.geo.matching`) ->
:func:`~somali_foodsec_radio.feedback.ipc_update.aggregate_weekly_impact` -> phase
adjustment.

Keyword lists, thresholds and phase effects live in ``config.yaml`` under ``feedback:``.
"""

from .ipc_update import adjust_ipc_phases_with_threshold, aggregate_weekly_impact
from .pdf_extract import clean_date, extract_detailed_calls_from_pdf, is_valid_date
from .signals import (
    SIGNAL_NAMES,
    create_impact_signals,
    infer_impact_level,
    signal_present,
)

__all__ = [
    "SIGNAL_NAMES",
    "adjust_ipc_phases_with_threshold",
    "aggregate_weekly_impact",
    "clean_date",
    "create_impact_signals",
    "extract_detailed_calls_from_pdf",
    "infer_impact_level",
    "is_valid_date",
    "signal_present",
]
