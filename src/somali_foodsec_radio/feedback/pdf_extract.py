"""Extract caller-feedback records from Radio Ergo weekly feedback PDFs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def is_valid_date(token: str) -> bool:
    """Return True if *token* looks like a ``D/M/YYYY`` or ``D-M-YYYY`` date."""
    return bool(re.match(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{4}$", token))


def clean_date(date_str: str) -> str:
    """Reformat a ``DD-MM-YYYY`` / ``DD/MM/YYYY`` date string to ``YYYY-MM-DD``.

    Returns the input unchanged if it cannot be parsed.
    """
    try:
        if "-" in date_str:
            day, month, year = map(int, date_str.split("-"))
        elif "/" in date_str:
            day, month, year = map(int, date_str.split("/"))
        else:
            return date_str
        return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:  # noqa: BLE001 - any parse failure -> return input unchanged
        return date_str


def extract_detailed_calls_from_pdf(pdf_path: Path) -> pd.DataFrame:
    """Extract call records from a Radio Ergo audience-feedback PDF.

    ``pdfplumber`` is imported lazily (it requires the ``[analysis]`` extras).

    Returns:
        DataFrame with columns ``['date', 'location', 'theme', 'remarks']``.
    """
    import pdfplumber

    call_records = []
    current_record = None
    inside_calls_section = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Detect the table header to start collecting.
                if (
                    not inside_calls_section
                    and "Date Location" in line
                    and "Theme" in line
                ):
                    inside_calls_section = True
                    continue

                if inside_calls_section:
                    tokens = line.split(maxsplit=4)

                    if len(tokens) < 5:
                        # Probably a continuation of the previous remark.
                        if current_record:
                            current_record["remarks"] += " " + line
                        continue

                    if is_valid_date(tokens[0]):
                        if current_record:
                            call_records.append(current_record)

                        date_raw, location, _gender, theme, remarks = tokens
                        current_record = {
                            "date": clean_date(date_raw),
                            "location": location.strip(),
                            "theme": theme.strip(),
                            "remarks": remarks.strip(),
                        }
                    else:
                        # Continuation line, not a new record.
                        if current_record:
                            current_record["remarks"] += " " + line

    if current_record:
        call_records.append(current_record)

    return pd.DataFrame(call_records)
