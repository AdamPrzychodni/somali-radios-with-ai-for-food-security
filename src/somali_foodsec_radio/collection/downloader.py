"""Download SoundCloud tracks to disk as MP3 files.

This downloader saves audio into ``data/raw/``. ``yt-dlp`` is imported lazily so the
module stays importable without the ``[asr]`` extras.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from ..config import get_setting
from ..logging_utils import get_logger
from ..paths import get_data_dir
from .scraper import collect_urls_in_date_range, create_session
from .urls import validate_soundcloud_url

logger = get_logger(__name__)


def get_yt_dlp_options(
    output_dir: Path | str,
    audio_quality: str = "192",
    archive_path: Path | str | None = None,
) -> dict:
    """Build the ``yt-dlp`` options dict for downloading audio as MP3.

    Args:
        output_dir: Directory to save downloaded files into.
        audio_quality: MP3 quality in kbps.
        archive_path: yt-dlp download-archive file; defaults to
            ``data/.ytdlp-archive.txt`` so completed downloads are skipped on re-runs.
    """
    if archive_path is None:
        archive_path = get_data_dir() / ".ytdlp-archive.txt"
    return {
        "format": "bestaudio/best",
        "outtmpl": str(Path(output_dir) / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": audio_quality,
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,  # continue past 404s and other download errors
        "overwrites": False,
        "download_archive": str(archive_path),
    }


def download_single_track(
    url: str, output_dir: Path | str, audio_quality: str = "192"
) -> str | None:
    """Download a single SoundCloud track; return the MP3 path or ``None`` on failure."""
    import yt_dlp

    if not validate_soundcloud_url(url):
        logger.error("Invalid SoundCloud URL: %s", url)
        return None

    ydl_opts = get_yt_dlp_options(output_dir, audio_quality)

    # Pre-check the expected filename so existing downloads are skipped.
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_probe = ydl.extract_info(url, download=False)
            if info_probe:
                expected = ydl.prepare_filename(info_probe)
                mp3_expected = f"{os.path.splitext(expected)[0]}.mp3"
                if Path(mp3_expected).exists():
                    logger.info("Skipping (already exists): %s", mp3_expected)
                    return mp3_expected
    except Exception as exc:  # noqa: BLE001 - precheck is best-effort
        logger.debug("Precheck failed for %s: %s", url, exc)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                logger.error(
                    "Download failed for %s: no info extracted "
                    "(likely a 404 or private track).",
                    url,
                )
                return None

            filename = ydl.prepare_filename(info)
            if not filename:
                logger.error("Download failed for %s: could not prepare filename.", url)
                return None

            base, _ = os.path.splitext(filename)
            mp3_file = f"{base}.mp3"
            if not Path(mp3_file).exists():
                logger.error("Download failed for %s: MP3 file not created.", url)
                return None

            logger.info("Successfully downloaded: %s", mp3_file)
            return mp3_file
    except Exception as exc:  # noqa: BLE001 - report and continue
        logger.error("Download failed for %s: %s", url, exc)
        return None


def download_multiple_tracks(
    urls: list[str], output_dir: Path | str, audio_quality: str = "192"
) -> tuple[list[str], list[str]]:
    """Download several tracks; return ``(downloaded_paths, failed_urls)``."""
    logger.info("Starting download of %d tracks to: %s", len(urls), output_dir)

    downloaded_files: list[str] = []
    failed_urls: list[str] = []

    for i, url in enumerate(urls, 1):
        logger.info("Downloading %d/%d: %s", i, len(urls), url)
        result = download_single_track(url, output_dir, audio_quality)
        if result:
            downloaded_files.append(result)
        else:
            failed_urls.append(url)
        time.sleep(1)  # rate limiting

    logger.info("%s", "=" * 60)
    logger.info("Download Summary:")
    logger.info("  Total URLs: %d", len(urls))
    logger.info("  Successful: %d", len(downloaded_files))
    logger.info("  Failed: %d", len(failed_urls))
    logger.info("%s", "=" * 60)
    for url in failed_urls:
        logger.warning("Failed URL: %s", url)

    return downloaded_files, failed_urls


class SoundCloudDownloader:
    """SoundCloud track downloader with date filtering.

    Handles URL validation, audio extraction and batch downloads. Files are saved to
    the project's ``data/raw/`` directory.
    """

    def __init__(self, audio_quality: str = "192"):
        """Args:
        audio_quality: MP3 quality in kbps.
        """
        self.audio_quality = audio_quality
        self.raw_data_dir = get_data_dir("raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.session = create_session()
        self._logged_output_dir = False
        logger.info("Initialized downloader. Base data directory: %s", self.raw_data_dir)

    def create_output_directory(self, dir_name: str | None = None) -> Path:
        """Return the ``data/raw/`` directory, ensuring it exists.

        All files are saved directly into ``data/raw/``; *dir_name* is accepted for
        backwards compatibility but ignored.
        """
        output_dir = self.raw_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self._logged_output_dir:
            logger.info("All files will be saved directly to: %s", output_dir)
            self._logged_output_dir = True
        return output_dir

    def download_by_date_range(
        self,
        profile_url: str,
        start_date: str,
        end_date: str,
        output_dir_name: str | None = None,
    ) -> list[str]:
        """Download tracks within ``[start_date, end_date]`` (dates as ``YYYY-MM-DD``).

        Raises:
            ValueError: If a date string is not ``YYYY-MM-DD``.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {exc}")

        output_dir = self.create_output_directory(output_dir_name)
        urls = collect_urls_in_date_range(profile_url, start_dt, end_dt, self.session)
        if not urls:
            logger.warning("No tracks found in the specified date range")
            return []

        downloaded_files, _ = download_multiple_tracks(
            urls, output_dir, self.audio_quality
        )
        return downloaded_files

    def download_urls(
        self, urls: list[str], output_dir_name: str | None = None
    ) -> list[str]:
        """Download a specific list of SoundCloud URLs."""
        output_dir = self.create_output_directory(output_dir_name)

        valid_urls = []
        for url in urls:
            if validate_soundcloud_url(url):
                valid_urls.append(url)
            else:
                logger.warning("Skipping invalid URL: %s", url)

        if not valid_urls:
            logger.warning("No valid URLs to download.")
            return []

        downloaded_files, _ = download_multiple_tracks(
            valid_urls, output_dir, self.audio_quality
        )
        return downloaded_files


def download_by_date_range(
    profile_url: str,
    start_date: str,
    end_date: str,
    output_dir: str | None = None,
    audio_quality: str = "192",
) -> list[str]:
    """Convenience wrapper: download tracks by date range into ``data/raw/``."""
    downloader = SoundCloudDownloader(audio_quality=audio_quality)
    return downloader.download_by_date_range(
        profile_url, start_date, end_date, output_dir
    )


def download_urls(
    urls: list[str], output_dir: str | None = None, audio_quality: str = "192"
) -> list[str]:
    """Convenience wrapper: download specific URLs into ``data/raw/``."""
    downloader = SoundCloudDownloader(audio_quality=audio_quality)
    return downloader.download_urls(urls, output_dir)


def download_radio_ergo_by_date(
    start_date: str, end_date: str, output_dir: str | None = None
) -> list[str]:
    """Convenience wrapper: download Radio Ergo broadcasts by date range.

    The Radio Ergo profile URL is read from ``config.yaml`` (``soundcloud.profile_url``).
    """
    profile_url = get_setting(
        "soundcloud.profile_url", "https://soundcloud.com/radio-ergo"
    )
    return download_by_date_range(profile_url, start_date, end_date, output_dir)
