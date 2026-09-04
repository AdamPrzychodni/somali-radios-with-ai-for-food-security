"""SoundCloud URL construction, validation and date parsing.

Pure functions — no network access — so they are cheap to import and easy to test.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from ..config import get_setting
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Month name -> number, covering both full names and three-letter abbreviations.
MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

SOUNDCLOUD_URL_PATTERN = r"^https?://(?:www\.)?soundcloud\.com/[\w-]+/[\w-]+"

# Recognised date forms in a track slug: ``DD-monthname-YYYY`` and ISO ``YYYY-MM-DD``.
DATE_PATTERNS = [
    r"(\d{1,2})-([a-z]+)-(\d{4})",
    r"(\d{4})-(\d{1,2})-(\d{1,2})",
]


def validate_soundcloud_url(url: str) -> bool:
    """Return True if *url* looks like a SoundCloud track URL (profile + track slug)."""
    return bool(re.match(SOUNDCLOUD_URL_PATTERN, url, re.IGNORECASE))


def extract_date_from_url(url: str) -> datetime | None:
    """Extract a broadcast date from a SoundCloud URL, or ``None`` if none is found.

    Recognises ``...-DD-monthname-YYYY`` (e.g. ``idaacadda-01-jul-2024``) and the ISO
    ``YYYY-MM-DD`` form.
    """
    url_path = url.split("/")[-1].lower()

    for pattern in DATE_PATTERNS:
        match = re.search(pattern, url_path, re.IGNORECASE)
        if not match or len(match.groups()) != 3:
            continue
        try:
            g1, g2, g3 = match.groups()
            if g2.isdigit():  # ISO form: YYYY-MM-DD
                year, month, day = int(g1), int(g2), int(g3)
            else:  # DD-monthname-YYYY
                day, year = int(g1), int(g3)
                month = MONTH_NAMES.get(g2.lower())
            if month and 1 <= month <= 12:
                return datetime(year, month, day)
        except (ValueError, TypeError) as exc:
            logger.debug("Date parsing failed for %s: %s", url, exc)
            continue
    return None


def extract_username_from_url(profile_url: str) -> str:
    """Return the username (last path segment) of a SoundCloud profile URL."""
    return profile_url.rstrip("/").split("/")[-1]


def generate_url_patterns(username: str, date: datetime) -> list[str]:
    """Generate candidate broadcast URLs for *date* (Radio-Ergo-style slug variants)."""
    slug = get_setting("soundcloud.broadcast_slug", "idaacadda")
    base_url = f"https://soundcloud.com/{username}"
    day = date.day
    month_full = date.strftime("%B").lower()
    month_abbr = date.strftime("%b").lower()
    year = date.year
    patterns = [
        f"{slug}-{day:02d}-{month_abbr}-{year}",
        f"{slug}-{day}-{month_abbr}-{year}",
        f"{slug}-{day:02d}-{month_full}-{year}",
        f"{slug}-{day}-{month_full}-{year}",
        f"show-{day:02d}-{month_abbr}-{year}",
        f"broadcast-{day}-{month_abbr}-{year}",
    ]
    return [f"{base_url}/{pattern}" for pattern in patterns]


def generate_urls_for_range(
    profile_url: str,
    start_date: datetime,
    end_date: datetime,
    slug: str = "idaacadda",
) -> list[tuple[datetime, str]]:
    """Generate one ``<profile>/<slug>-DD-mon-YYYY`` URL per day in the range.

    Returns a list of ``(date, url)`` tuples covering *start_date*..*end_date* inclusive.
    """
    urls: list[tuple[datetime, str]] = []
    profile_url = profile_url.rstrip("/")
    current_date = start_date
    while current_date <= end_date:
        day = current_date.day
        month = current_date.strftime("%b").lower()
        year = current_date.year
        urls.append((current_date, f"{profile_url}/{slug}-{day:02d}-{month}-{year}"))
        current_date += timedelta(days=1)
    return urls
