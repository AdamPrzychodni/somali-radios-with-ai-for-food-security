"""Scrape Radio Ergo's SoundCloud profile for broadcast track URLs."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..logging_utils import get_logger
from .urls import (
    extract_date_from_url,
    extract_username_from_url,
    generate_url_patterns,
    validate_soundcloud_url,
)

logger = get_logger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


def create_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    """Return a requests session with a browser-like ``User-Agent`` header."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_profile_tracks(profile_url: str, session: requests.Session) -> list[str]:
    """Fetch all unique track URLs linked from a SoundCloud profile page."""
    try:
        response = session.get(profile_url, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network failure should not crash collection
        logger.error("Failed to fetch profile %s: %s", profile_url, exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    username = extract_username_from_url(profile_url)
    track_urls: list[str] = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Keep track URLs only: same user, not a playlist, deep enough to be a track.
        if not (
            href.startswith("/")
            and username in href
            and "/sets/" not in href
            and href.count("/") >= 2
        ):
            continue
        full_url = urljoin("https://soundcloud.com", href)
        if validate_soundcloud_url(full_url):
            track_urls.append(full_url)

    return list(dict.fromkeys(track_urls))  # de-duplicate, preserve order


def check_url_exists(url: str, session: requests.Session) -> bool:
    """Return True if *url* appears to be a real SoundCloud track page.

    A bare HEAD request is unreliable (SoundCloud answers 200 for many non-track
    pages), so this fetches the page and looks for track-specific markup.
    """
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return False

        soup = BeautifulSoup(response.text, "html.parser")
        meta_tag = soup.find(
            "meta", attrs={"property": "og:type", "content": "music.song"}
        )
        if meta_tag:
            return True

        # Less reliable fallback.
        if soup.find("a", class_=re.compile(r"playButton")):
            return True

        logger.debug("URL %s is 200 OK but not a valid track page.", url)
        return False
    except Exception as exc:  # noqa: BLE001 - treat any failure as "does not exist"
        logger.debug("Failed to check URL %s: %s", url, exc)
        return False


def collect_urls_in_date_range(
    profile_url: str,
    start_date: datetime,
    end_date: datetime,
    session: requests.Session,
) -> list[str]:
    """Collect broadcast URLs within a date range.

    Combines two strategies: parsing tracks linked on the profile page, and
    generating + probing candidate URLs from known date-slug patterns.
    """
    logger.info("Searching for tracks from %s to %s", start_date.date(), end_date.date())

    # Strategy 1: tracks already linked on the profile page.
    profile_tracks = fetch_profile_tracks(profile_url, session)
    urls_found: list[str] = []
    for url in profile_tracks:
        track_date = extract_date_from_url(url)
        if track_date and start_date <= track_date <= end_date:
            urls_found.append(url)
            logger.info("Found matching URL (from profile): %s", url)

    # Strategy 2: probe candidate URLs generated from date-slug patterns.
    logger.info("Checking for additional URLs using date patterns...")
    username = extract_username_from_url(profile_url)
    current_date = start_date
    while current_date <= end_date:
        for url in generate_url_patterns(username, current_date):
            if url not in urls_found and check_url_exists(url, session):
                urls_found.append(url)
                logger.info("Found additional URL (from pattern): %s", url)
                time.sleep(0.5)  # rate limiting
        current_date += timedelta(days=1)

    return list(dict.fromkeys(urls_found))
