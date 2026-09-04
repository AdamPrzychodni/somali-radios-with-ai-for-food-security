"""Streaming SoundCloud collection: download to memory, transcribe, never stage to disk.

Audio is downloaded into an in-memory buffer, transcribed with the Somali Wav2Vec2
engine, and discarded — only transcripts are persisted. ``yt-dlp`` and the ASR engine
are imported lazily so this module stays importable without the ``[asr]`` extras.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..logging_utils import SilentLogger
from .urls import generate_urls_for_range

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _output_filenames() -> dict:
    """Return the CSV/log filenames from config, with safe defaults."""
    try:
        from ..config import get_config

        outputs = get_config().get("outputs", {})
    except Exception:  # noqa: BLE001 - fall back to defaults if config is unavailable
        outputs = {}
    return {
        "transcriptions_csv": outputs.get(
            "transcriptions_csv", "transcriptions_database.csv"
        ),
        "transcription_log": outputs.get(
            "transcription_log", ".transcription_log.json"
        ),
    }


class StreamingSoundCloudDownloader:
    """GPU-optimised batch processor for SoundCloud audio."""

    def __init__(
        self,
        output_dir: str,
        output_format: str = "structured",
        batch_size: int = 8,
        use_fp16: bool = True,
        verbose: bool = False,
        transcription_engine: object | None = None,
    ):
        """Initialise the downloader.

        Args:
            output_dir: Directory for outputs.
            output_format: ``'structured'`` (CSV), ``'txt'``, or ``'both'``.
            batch_size: GPU batch size for inference.
            use_fp16: Enable FP16 mixed precision.
            verbose: Enable detailed logging.
            transcription_engine: Optional pre-built ASR engine. When ``None`` a
                :class:`SomaliASREngine` is created lazily (requires the ``[asr]``
                extras).
        """
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.verbose = verbose
        self.ffmpeg_path = self._find_ffmpeg()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        filenames = _output_filenames()
        self.log_file = self.output_dir / filenames["transcription_log"]
        self.structured_data_file = self.output_dir / filenames["transcriptions_csv"]

        self.transcription_log = self._load_transcription_log()
        self._init_structured_storage()

        if transcription_engine is not None:
            self.transcription_engine = transcription_engine
        else:
            from ..transcription.asr_wav2vec2 import SomaliASREngine

            self.transcription_engine = SomaliASREngine(
                batch_size=batch_size, use_fp16=use_fp16, verbose=verbose
            )

        if verbose:
            vram = self.transcription_engine.get_vram_usage()
            print(
                f"Initial VRAM: {vram['allocated_gb']:.2f} GB allocated, "
                f"{vram['reserved_gb']:.2f} GB reserved"
            )

    def _find_ffmpeg(self) -> str | None:
        """Locate the ffmpeg binary."""
        if self.verbose:
            print("Checking for ffmpeg...")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            if self.verbose:
                print(f"✓ Found ffmpeg: {ffmpeg_path}")
            return ffmpeg_path

        for location in (
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
        ):
            if Path(location).exists():
                if self.verbose:
                    print(f"✓ Found ffmpeg: {location}")
                return location

        if self.verbose:
            print("⚠️ ffmpeg not found")
        return None

    def _init_structured_storage(self) -> None:
        """Create the CSV database with headers if it does not exist."""
        if not self.structured_data_file.exists():
            columns = [
                "id",
                "url",
                "title",
                "date_recorded",
                "date_processed",
                "processing_duration_seconds",
                "audio_size_mb",
                "audio_duration_seconds",
                "transcript_length_chars",
                "transcript_length_words",
                "transcript_text",
            ]
            pd.DataFrame(columns=columns).to_csv(self.structured_data_file, index=False)

    def _load_transcription_log(self) -> dict:
        """Load the processing log used to skip already-processed URLs."""
        if self.log_file.exists():
            try:
                with open(self.log_file) as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_transcription_log(self) -> None:
        """Persist the processing log."""
        with open(self.log_file, "w") as fh:
            json.dump(self.transcription_log, fh, indent=2)

    def _download_to_memory(self, url: str) -> tuple[bytes | None, dict]:
        """Download audio for *url* into an in-memory buffer."""
        import yt_dlp
        from yt_dlp.utils import DownloadError

        metadata: dict = {"success": False, "error": None, "info": {}}

        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "logger": SilentLogger(),
        }
        if self.ffmpeg_path:
            ydl_opts["ffmpeg_location"] = self.ffmpeg_path

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                info = info or {}
                metadata["info"] = {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "upload_date": info.get("upload_date", ""),
                }

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                ydl_opts["outtmpl"] = tmp_file.name.replace(".mp3", ".%(ext)s")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                processed_file = tmp_file.name.replace(".mp3", ".mp3")

                if Path(processed_file).exists():
                    with open(processed_file, "rb") as fh:
                        audio_data = fh.read()
                    Path(processed_file).unlink()
                    metadata["success"] = True
                    return audio_data, metadata
        except DownloadError as exc:
            if "HTTP Error 404" in str(exc):
                metadata["error"] = "Broadcast not found (404 Error)."
            else:
                metadata["error"] = f"Download failed: {exc}"
            return None, metadata
        except Exception as exc:  # noqa: BLE001 - report and continue
            metadata["error"] = str(exc)
            return None, metadata

        return None, metadata

    def _transcribe_audio_data(
        self, audio_data: bytes, metadata: dict
    ) -> tuple[str | None, dict]:
        """GPU-accelerated transcription of in-memory audio."""
        start_time = time.time()

        if not audio_data:
            metadata["transcription_error"] = "Empty audio data"
            return None, metadata

        try:
            metadata["audio_size_mb"] = len(audio_data) / (1024 * 1024)

            if PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                metadata["audio_duration_seconds"] = len(audio_segment) / 1000.0

            transcript = self.transcription_engine.transcribe_from_memory_batched(
                audio_data
            )
            vram = self.transcription_engine.get_vram_usage()

            metadata.update(
                {
                    "processing_duration_seconds": time.time() - start_time,
                    "transcript_length_chars": len(transcript),
                    "transcript_length_words": len(transcript.split()),
                    "transcription_success": True,
                    "vram_used_gb": vram["allocated_gb"],
                }
            )
            return transcript, metadata
        except Exception as exc:  # noqa: BLE001 - record failure metadata
            metadata.update(
                {
                    "transcription_error": str(exc),
                    "transcription_success": False,
                    "processing_duration_seconds": time.time() - start_time,
                }
            )
            return None, metadata

    def _save_structured_data(self, record: dict) -> None:
        """Append one record to the CSV database."""
        try:
            new_row = pd.DataFrame([record])
            new_row.to_csv(
                self.structured_data_file, mode="a", header=False, index=False
            )
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"⚠️ Could not save to database: {exc}")

    def _save_transcript_file(
        self, transcript: str, metadata: dict, base_filename: str
    ) -> str:
        """Save a transcript as a human-readable text file."""
        transcript_file = self.output_dir / f"{base_filename}.txt"
        with open(transcript_file, "w", encoding="utf-8") as fh:
            fh.write(f"# Transcription Report\n{'=' * 50}\n")
            fh.write(f"Source URL: {metadata.get('url', 'Unknown')}\n")
            fh.write(f"Title: {metadata.get('info', {}).get('title', 'Unknown')}\n")
            fh.write(
                f"Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            fh.write("Model: Mustafaa4a/ASR-Somali (GPU-optimized)\n")
            fh.write(
                f"Audio Duration: {metadata.get('audio_duration_seconds', 0):.1f}s\n"
            )
            fh.write(
                "Processing Time: "
                f"{metadata.get('processing_duration_seconds', 0):.1f}s\n"
            )
            fh.write(f"VRAM Used: {metadata.get('vram_used_gb', 0):.2f} GB\n")
            fh.write(f"Word Count: {metadata.get('transcript_length_words', 0)}\n")
            fh.write(f"{'=' * 50}\n\n## Transcript\n\n{transcript}")
        return str(transcript_file)

    def _process_url(self, url: str) -> tuple[bool, str | None, dict]:
        """Download, transcribe and persist a single URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()

        if url_hash in self.transcription_log:
            entry = self.transcription_log[url_hash]
            return True, entry.get("file_path"), entry

        audio_data, metadata = self._download_to_memory(url)
        if not audio_data:
            return False, None, metadata

        metadata["url"] = url
        metadata["id"] = url_hash

        transcript, metadata = self._transcribe_audio_data(audio_data, metadata)
        if not transcript:
            return False, None, metadata

        safe_title = re.sub(r"[^\w\s-]", "", metadata.get("info", {}).get("title", ""))
        base_filename = (
            re.sub(r"[-\s]+", "-", safe_title).strip("-")
            or f"soundcloud_{url_hash[:8]}"
        )

        file_path: str | None = None

        if self.output_format in ("structured", "both"):
            record = {
                "id": url_hash,
                "url": url,
                "title": metadata.get("info", {}).get("title", "Unknown"),
                "date_recorded": metadata.get("info", {}).get("upload_date", ""),
                "date_processed": datetime.now().isoformat(),
                "processing_duration_seconds": metadata.get(
                    "processing_duration_seconds", 0
                ),
                "audio_size_mb": metadata.get("audio_size_mb", 0),
                "audio_duration_seconds": metadata.get("audio_duration_seconds", 0),
                "transcript_length_chars": metadata.get("transcript_length_chars", 0),
                "transcript_length_words": metadata.get("transcript_length_words", 0),
                "transcript_text": transcript,
            }
            self._save_structured_data(record)
            file_path = str(self.structured_data_file)

        if self.output_format in ("txt", "both"):
            file_path = self._save_transcript_file(transcript, metadata, base_filename)

        self.transcription_log[url_hash] = {
            "url": url,
            "title": metadata.get("info", {}).get("title", "Unknown"),
            "file_path": file_path,
            "success": True,
        }
        self._save_transcription_log()

        return True, file_path, metadata

    def process_date_range(
        self, profile_url: str, start_date_str: str, end_date_str: str
    ) -> dict:
        """Process every broadcast in a date range with GPU acceleration."""
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD'") from exc

        print(f"\n{'=' * 70}")
        print("GPU-Optimized SoundCloud ASR Pipeline")
        print(f"{'=' * 70}")
        print(f"Profile: {profile_url}")
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Device: {self.transcription_engine.device.upper()}")
        print(f"Batch Size: {self.transcription_engine.batch_size}")
        print(f"Output: {self.structured_data_file}")
        print(f"{'=' * 70}\n")

        urls = generate_urls_for_range(profile_url, start_date, end_date)

        if not urls:
            print("No URLs generated for date range")
            return {"successful": [], "failed": [], "skipped": []}

        results: dict[str, list] = {"successful": [], "failed": [], "skipped": []}

        with tqdm(urls, desc="Processing tracks", unit="track") as pbar:
            for date, url in pbar:
                pbar.set_postfix_str(f"{date.date()}")

                try:
                    success, _, metadata = self._process_url(url)
                    if success:
                        results["successful"].append(url)
                        if (
                            "processing_duration_seconds" in metadata
                            and "audio_duration_seconds" in metadata
                        ):
                            speedup = metadata["audio_duration_seconds"] / max(
                                metadata["processing_duration_seconds"], 0.1
                            )
                            pbar.set_postfix_str(
                                f"{date.date()} | {speedup:.1f}x realtime"
                            )
                    else:
                        results["failed"].append(url)
                        error_msg = metadata.get("error")
                        if error_msg and "Broadcast not found" in error_msg:
                            tqdm.write(
                                f"↪️  Skipping {date.date()}: Broadcast not found."
                            )
                        elif self.verbose and error_msg:
                            tqdm.write(f"✗ Error on {date.date()}: {error_msg}")
                except Exception as exc:  # noqa: BLE001
                    if self.verbose:
                        tqdm.write(f"✗ Error on {date.date()}: {exc}")
                    results["failed"].append(url)

                time.sleep(0.3)  # short pause; processing is the bottleneck

        print(f"\n{'=' * 70}")
        print("Processing Complete")
        print(f"{'=' * 70}")
        print(f"✓ Successful: {len(results['successful'])}")
        print(f"✗ Failed: {len(results['failed'])}")

        vram = self.transcription_engine.get_vram_usage()
        print(f"Final VRAM: {vram['allocated_gb']:.2f} GB allocated")
        print(f"Database: {self.structured_data_file}")
        print(f"{'=' * 70}\n")

        return results


def run_batch_collection(
    profile_url: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    batch_size_days: int = 30,
    gpu_batch_size: int = 8,
    use_fp16: bool = True,
    verbose: bool = False,
) -> None:
    """GPU-optimised batch collection with crash recovery.

    Args:
        profile_url: SoundCloud profile URL.
        start_date: Start date ``YYYY-MM-DD``.
        end_date: End date ``YYYY-MM-DD``.
        output_dir: Output directory path.
        batch_size_days: Process the range in chunks of N days.
        gpu_batch_size: GPU inference batch size (higher = more VRAM, faster).
        use_fp16: Enable FP16 mixed precision (recommended for L4).
        verbose: Enable detailed logging.
    """
    print(f"\n{'=' * 80}")
    print("GPU-OPTIMIZED PRODUCTION DATA COLLECTION")
    print(f"{'=' * 80}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Batch Size: {batch_size_days} days")
    print(f"GPU Batch Size: {gpu_batch_size}")
    print(f"FP16 Precision: {use_fp16}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}\n")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    estimated_minutes = total_days * 3 / 4  # ~45 seconds per track on GPU

    print(f"📊 Total Scope: {total_days} days")
    print(
        f"📦 Estimated Batches: {(total_days + batch_size_days - 1) // batch_size_days}"
    )
    print(f"⏱️  Estimated Time: ~{estimated_minutes / 60:.1f} hours (GPU-accelerated)\n")

    current_start = start
    batch_num = 1
    total_successful = 0
    total_failed = 0

    while current_start <= end:
        batch_end = min(current_start + timedelta(days=batch_size_days - 1), end)

        print(f"\n{'─' * 80}")
        print(f"🔄 BATCH {batch_num}: {current_start.date()} → {batch_end.date()}")
        print(f"{'─' * 80}")

        try:
            downloader = StreamingSoundCloudDownloader(
                output_dir=output_dir,
                output_format="structured",
                batch_size=gpu_batch_size,
                use_fp16=use_fp16,
                verbose=verbose,
            )
            results = downloader.process_date_range(
                profile_url=profile_url,
                start_date_str=current_start.strftime("%Y-%m-%d"),
                end_date_str=batch_end.strftime("%Y-%m-%d"),
            )
            total_successful += len(results["successful"])
            total_failed += len(results["failed"])
            print(
                f"✓ Batch {batch_num} complete: "
                f"{len(results['successful'])} successful, "
                f"{len(results['failed'])} failed"
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  INTERRUPTED by user")
            print(f"📍 Progress saved up to: {current_start.date()}")
            print("💡 Resume by re-running with the same parameters")
            break
        except Exception as exc:  # noqa: BLE001 - keep going to the next batch
            print(f"\n❌ ERROR in batch {batch_num}: {exc}")
            print("📍 Progress saved. Continuing to next batch...")

        current_start = batch_end + timedelta(days=1)
        batch_num += 1

    print(f"\n\n{'=' * 80}")
    print("📊 COLLECTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Successful: {total_successful}")
    print(f"Total Failed: {total_failed}")
    print(f"Database: {Path(output_dir) / 'transcriptions_database.csv'}")
    print(f"{'=' * 80}\n")
