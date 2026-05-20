"""Download Radio Ergo broadcasts from SoundCloud.

Two download strategies are available:

* :class:`~somali_foodsec_radio.collection.downloader.SoundCloudDownloader` saves MP3
  files to disk (``data/raw/``).
* :class:`~somali_foodsec_radio.collection.streaming.StreamingSoundCloudDownloader`
  downloads to memory and transcribes on the fly, never staging audio on disk.
"""

from .downloader import (
    SoundCloudDownloader,
    download_by_date_range,
    download_radio_ergo_by_date,
    download_urls,
)
from .streaming import StreamingSoundCloudDownloader, run_batch_collection
from .urls import (
    extract_date_from_url,
    generate_urls_for_range,
    validate_soundcloud_url,
)

__all__ = [
    "SoundCloudDownloader",
    "StreamingSoundCloudDownloader",
    "download_by_date_range",
    "download_radio_ergo_by_date",
    "download_urls",
    "extract_date_from_url",
    "generate_urls_for_range",
    "run_batch_collection",
    "validate_soundcloud_url",
]
