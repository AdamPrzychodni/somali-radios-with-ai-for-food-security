"""
Production-ready SoundCloud ASR Transcription Pipeline.
Self-contained batch processor with crash recovery and resumption support.
Designed for multi-year data collection runs (2020-2025).
"""

import os
import re
import hashlib
import json
import time
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import io

import yt_dlp
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from tqdm import tqdm

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ============================================================================
# ASR ENGINE
# ============================================================================

class SomaliASREngine:
    """Transcription engine using Mustafaa4a/ASR-Somali model."""
    
    def __init__(self, model_name: str = "Mustafaa4a/ASR-Somali", verbose: bool = False):
        """
        Initialize Somali ASR model.
        
        Args:
            model_name: HuggingFace model identifier
            verbose: If True, print detailed loading information
        """
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.verbose = verbose
        self._setup_model(model_name)

    def _setup_model(self, model_name: str) -> None:
        """
        Load Wav2Vec2 model and processor.
        
        Raises:
            RuntimeError: If model fails to load
        """
        if self.verbose:
            print(f"Loading ASR model: {model_name}")
            
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            if self.verbose:
                print(f"✓ Model loaded on {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model: {e}")

    def transcribe_from_memory(self, audio_data: bytes) -> str:
        """
        Transcribe audio from memory buffer.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcribed text
            
        Raises:
            ValueError: If transcription fails
        """
        target_sr = 16000
        
        try:
            # Load and resample audio
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=target_sr)
            
            # Adaptive chunk sizing based on device
            chunk_length_s = 45 if self.device == 'cuda' else 20
            chunk_length = chunk_length_s * target_sr
            
            transcriptions = []
            
            # Process in chunks
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                
                input_values = self.processor(
                    chunk,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                transcriptions.append(transcription)
            
            return " ".join(transcriptions).strip()

        except Exception as e:
            raise ValueError(f"Transcription failed: {e}")


# ============================================================================
# DOWNLOADER & PROCESSOR
# ============================================================================

class StreamingSoundCloudDownloader:
    """
    Batch processor for SoundCloud audio with Somali ASR transcription.
    Optimized for long date ranges with minimal metadata overhead.
    """
    
    def __init__(self, output_dir: str, output_format: str = "structured", verbose: bool = False):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory for outputs
            output_format: 'structured' (CSV), 'txt', or 'both'
            verbose: Enable detailed logging
        """
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.verbose = verbose
        self.ffmpeg_path = self._find_ffmpeg()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.output_dir / ".transcription_log.json"
        self.structured_data_file = self.output_dir / "transcriptions_database.csv"
        
        self.transcription_log = self._load_transcription_log()
        self._init_structured_storage()
        
        self.transcription_engine = SomaliASREngine(verbose=verbose)

    def _find_ffmpeg(self) -> Optional[str]:
        """Locate ffmpeg binary."""
        if self.verbose:
            print("Checking for ffmpeg...")
            
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            if self.verbose:
                print(f"✓ Found ffmpeg: {ffmpeg_path}")
            return ffmpeg_path
        
        # Check common locations
        for location in ['/usr/bin/ffmpeg', '/home/zeus/miniconda3/bin/ffmpeg']:
            if Path(location).exists():
                if self.verbose:
                    print(f"✓ Found ffmpeg: {location}")
                return location
                
        if self.verbose:
            print("⚠️ ffmpeg not found")
        return None

    def _init_structured_storage(self) -> None:
        """Initialize CSV database with minimal essential columns."""
        if not self.structured_data_file.exists():
            columns = [
                'id',
                'url',
                'title',
                'date_recorded',
                'date_processed',
                'processing_duration_seconds',
                'audio_size_mb',
                'audio_duration_seconds',
                'transcript_length_chars',
                'transcript_length_words',
                'transcript_text'
            ]
            pd.DataFrame(columns=columns).to_csv(self.structured_data_file, index=False)

    def _load_transcription_log(self) -> Dict:
        """Load processing log to skip already-processed URLs."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_transcription_log(self) -> None:
        """Persist processing log."""
        with open(self.log_file, 'w') as f:
            json.dump(self.transcription_log, f, indent=2)

    def _download_to_memory(self, url: str) -> Tuple[Optional[bytes], Dict]:
        """
        Download audio to memory buffer.
        
        Args:
            url: SoundCloud track URL
            
        Returns:
            Tuple of (audio_bytes, metadata_dict)
        """
        metadata = {'success': False, 'error': None, 'info': {}}
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        }
        
        if self.ffmpeg_path:
            ydl_opts['ffmpeg_location'] = self.ffmpeg_path

        try:
            # Extract metadata
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                metadata['info'] = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                }
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                ydl_opts['outtmpl'] = tmp_file.name.replace('.mp3', '.%(ext)s')
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                processed_file = tmp_file.name.replace('.mp3', '.mp3')
                
                if Path(processed_file).exists():
                    with open(processed_file, 'rb') as f:
                        audio_data = f.read()
                    
                    Path(processed_file).unlink()
                    metadata['success'] = True
                    return audio_data, metadata
                    
        except Exception as e:
            metadata['error'] = str(e)
            return None, metadata
            
        return None, metadata

    def _transcribe_audio_data(self, audio_data: bytes, metadata: Dict) -> Tuple[Optional[str], Dict]:
        """
        Transcribe audio from memory.
        
        Args:
            audio_data: Raw audio bytes
            metadata: Metadata dictionary to update
            
        Returns:
            Tuple of (transcript_text, updated_metadata)
        """
        start_time = time.time()
        
        if not audio_data:
            metadata['transcription_error'] = "Empty audio data"
            return None, metadata
            
        try:
            metadata['audio_size_mb'] = len(audio_data) / (1024 * 1024)
            
            if PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                metadata['audio_duration_seconds'] = len(audio_segment) / 1000.0

            transcript = self.transcription_engine.transcribe_from_memory(audio_data)
            
            metadata.update({
                'processing_duration_seconds': time.time() - start_time,
                'transcript_length_chars': len(transcript),
                'transcript_length_words': len(transcript.split()),
                'transcription_success': True,
            })
            
            return transcript, metadata
            
        except Exception as e:
            metadata.update({
                'transcription_error': str(e),
                'transcription_success': False,
                'processing_duration_seconds': time.time() - start_time,
            })
            return None, metadata

    def _save_structured_data(self, record: Dict) -> None:
        """Append record to CSV database."""
        try:
            new_row = pd.DataFrame([record])
            new_row.to_csv(self.structured_data_file, mode='a', header=False, index=False)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not save to database: {e}")

    def _save_transcript_file(self, transcript: str, metadata: Dict, base_filename: str) -> str:
        """Save transcript as text file with metadata header."""
        transcript_file = self.output_dir / f"{base_filename}.txt"
        
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"# Transcription Report\n{'='*50}\n")
            f.write(f"Source URL: {metadata.get('url', 'Unknown')}\n")
            f.write(f"Title: {metadata.get('info', {}).get('title', 'Unknown')}\n")
            f.write(f"Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Mustafaa4a/ASR-Somali\n")
            f.write(f"Audio Duration: {metadata.get('audio_duration_seconds', 0):.1f}s\n")
            f.write(f"Processing Time: {metadata.get('processing_duration_seconds', 0):.1f}s\n")
            f.write(f"Word Count: {metadata.get('transcript_length_words', 0)}\n")
            f.write(f"{'='*50}\n\n## Transcript\n\n{transcript}")
            
        return str(transcript_file)

    def _process_url(self, url: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Process single URL: download, transcribe, save.
        
        Args:
            url: SoundCloud track URL
            
        Returns:
            Tuple of (success, file_path, metadata)
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Skip if already processed (CRITICAL for resumption)
        if url_hash in self.transcription_log:
            return True, self.transcription_log[url_hash].get('file_path'), self.transcription_log[url_hash]
        
        # Download
        audio_data, metadata = self._download_to_memory(url)
        if not audio_data:
            return False, None, metadata
            
        metadata['url'] = url
        metadata['id'] = url_hash
        
        # Transcribe
        transcript, metadata = self._transcribe_audio_data(audio_data, metadata)
        if not transcript:
            return False, None, metadata
        
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', metadata.get('info', {}).get('title', ''))
        base_filename = re.sub(r'[-\s]+', '-', safe_title).strip('-') or f"soundcloud_{url_hash[:8]}"

        file_path = None
        
        # Save to database
        if self.output_format in ['structured', 'both']:
            record = {
                'id': url_hash,
                'url': url,
                'title': metadata.get('info', {}).get('title', 'Unknown'),
                'date_recorded': metadata.get('info', {}).get('upload_date', ''),
                'date_processed': datetime.now().isoformat(),
                'processing_duration_seconds': metadata.get('processing_duration_seconds', 0),
                'audio_size_mb': metadata.get('audio_size_mb', 0),
                'audio_duration_seconds': metadata.get('audio_duration_seconds', 0),
                'transcript_length_chars': metadata.get('transcript_length_chars', 0),
                'transcript_length_words': metadata.get('transcript_length_words', 0),
                'transcript_text': transcript
            }
            self._save_structured_data(record)
            file_path = str(self.structured_data_file)
        
        # Save text file
        if self.output_format in ['txt', 'both']:
            txt_file = self._save_transcript_file(transcript, metadata, base_filename)
            file_path = txt_file
        
        # Update log (enables resumption)
        self.transcription_log[url_hash] = {
            'url': url,
            'title': metadata.get('info', {}).get('title', 'Unknown'),
            'file_path': file_path,
            'success': True
        }
        self._save_transcription_log()
        
        return True, file_path, metadata

    def _generate_urls_for_range(self, profile_url: str, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, str]]:
        """
        Generate SoundCloud URLs for date range.
        
        Args:
            profile_url: Base SoundCloud profile URL
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of (date, url) tuples
        """
        urls = []
        profile_url = profile_url.rstrip('/')
        current_date = start_date
        
        while current_date <= end_date:
            day = current_date.day
            month = current_date.strftime('%b').lower()
            year = current_date.year
            
            url = f"{profile_url}/idaacadda-{day:02d}-{month}-{year}"
            urls.append((current_date, url))
            current_date += timedelta(days=1)
            
        return urls

    def process_date_range(self, profile_url: str, start_date_str: str, end_date_str: str) -> Dict:
        """
        Process all tracks for a date range with progress tracking.
        
        Args:
            profile_url: SoundCloud profile URL
            start_date_str: Start date in 'YYYY-MM-DD' format
            end_date_str: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with 'successful', 'failed', 'skipped' URL lists
            
        Raises:
            ValueError: If date format is invalid
        """
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD'")

        print(f"\n{'='*70}")
        print(f"SoundCloud ASR Transcription Pipeline")
        print(f"{'='*70}")
        print(f"Profile: {profile_url}")
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Output: {self.structured_data_file}")
        print(f"{'='*70}\n")
        
        urls = self._generate_urls_for_range(profile_url, start_date, end_date)
        
        if not urls:
            print("No URLs generated for date range")
            return {'successful': [], 'failed': [], 'skipped': []}
        
        results = {'successful': [], 'failed': [], 'skipped': []}
        
        # Process with progress bar
        with tqdm(urls, desc="Processing tracks", unit="track") as pbar:
            for date, url in pbar:
                pbar.set_postfix_str(f"{date.date()}")
                
                try:
                    success, _, _ = self._process_url(url)
                    if success:
                        results['successful'].append(url)
                    else:
                        results['failed'].append(url)
                except Exception as e:
                    if self.verbose:
                        tqdm.write(f"✗ Error on {date.date()}: {e}")
                    results['failed'].append(url)
                
                time.sleep(0.5)  # Rate limiting

        # Summary
        print(f"\n{'='*70}")
        print("Processing Complete")
        print(f"{'='*70}")
        print(f"✓ Successful: {len(results['successful'])}")
        print(f"✗ Failed: {len(results['failed'])}")
        print(f"Database: {self.structured_data_file}")
        
        if results['failed'] and self.verbose:
            print("\nFailed URLs:")
            for url in results['failed']:
                print(f"  - {url}")
        
        print(f"{'='*70}\n")
        
        return results


# ============================================================================
# BATCH PROCESSING WITH RESUMPTION
# ============================================================================

def run_batch_collection(
    profile_url: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    batch_size_days: int = 30,
    verbose: bool = False
) -> None:
    """
    Process large date ranges in resumable batches with crash recovery.
    
    Args:
        profile_url: SoundCloud profile URL
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        output_dir: Output directory path
        batch_size_days: Process in chunks of N days (default: 30)
        verbose: Enable detailed logging
        
    Example:
        run_batch_collection(
            profile_url="https://soundcloud.com/radio-ergo",
            start_date="2020-01-01",
            end_date="2025-09-30",
            output_dir="./data/02_intermediate/transcripts/mustafaa4a_ASR-Somali",
            batch_size_days=30
        )
    """
    print(f"\n{'='*80}")
    print("PRODUCTION DATA COLLECTION - BATCH PROCESSOR")
    print(f"{'='*80}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Batch Size: {batch_size_days} days")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1
    
    print(f"📊 Total Scope: {total_days} days")
    print(f"📦 Estimated Batches: {(total_days + batch_size_days - 1) // batch_size_days}")
    print(f"⏱️  Estimated Time: ~{total_days * 3 / 60:.1f} hours (at 3 min/track)\n")
    
    current_start = start
    batch_num = 1
    total_successful = 0
    total_failed = 0
    
    # Process in batches
    while current_start <= end:
        # Calculate batch end (don't exceed final end date)
        batch_end = min(
            current_start + timedelta(days=batch_size_days - 1),
            end
        )
        
        print(f"\n{'─'*80}")
        print(f"🔄 BATCH {batch_num}: {current_start.date()} → {batch_end.date()}")
        print(f"{'─'*80}")
        
        try:
            # Process batch (auto-skips already processed URLs via log)
            downloader = StreamingSoundCloudDownloader(
                output_dir=output_dir,
                output_format="structured",
                verbose=verbose
            )
            
            results = downloader.process_date_range(
                profile_url=profile_url,
                start_date_str=current_start.strftime('%Y-%m-%d'),
                end_date_str=batch_end.strftime('%Y-%m-%d')
            )
            
            total_successful += len(results['successful'])
            total_failed += len(results['failed'])
            
            print(f"✓ Batch {batch_num} complete: {len(results['successful'])} successful, {len(results['failed'])} failed")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  INTERRUPTED by user")
            print(f"📍 Progress saved up to: {current_start.date()}")
            print(f"💡 Resume by re-running with same parameters")
            print(f"   Already processed URLs will be automatically skipped")
            break
            
        except Exception as e:
            print(f"\n❌ ERROR in batch {batch_num}: {e}")
            print(f"📍 Progress saved. Continuing to next batch...")
            # Continue to next batch rather than failing entire run
        
        # Move to next batch
        current_start = batch_end + timedelta(days=1)
        batch_num += 1
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("📊 COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Successful: {total_successful}")
    print(f"Total Failed: {total_failed}")
    print(f"Database: {Path(output_dir) / 'transcriptions_database.csv'}")
    print(f"Log: {Path(output_dir) / '.transcription_log.json'}")
    print(f"{'='*80}\n")


def check_progress(output_dir: str) -> None:
    """
    Display current collection progress from existing logs.
    
    Args:
        output_dir: Output directory containing logs
    """
    output_path = Path(output_dir)
    log_file = output_path / ".transcription_log.json"
    csv_file = output_path / "transcriptions_database.csv"
    
    print(f"\n{'='*80}")
    print("COLLECTION PROGRESS REPORT")
    print(f"{'='*80}\n")
    
    # Check log
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        print(f"📝 Transcription Log:")
        print(f"   Total URLs Processed: {len(log_data)}")
        
        successful = sum(1 for v in log_data.values() if v.get('success', False))
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(log_data) - successful}")
    else:
        print("⚠️  No transcription log found")
    
    # Check CSV database
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"\n📊 Database Statistics:")
        print(f"   Total Records: {len(df)}")
        
        if not df.empty and 'date_recorded' in df.columns:
            df['date_recorded'] = pd.to_datetime(df['date_recorded'], format='%Y%m%d', errors='coerce')
            date_range = df['date_recorded'].agg(['min', 'max'])
            print(f"   Date Range: {date_range['min'].date()} to {date_range['max'].date()}")
        
        if 'audio_duration_seconds' in df.columns:
            total_audio = df['audio_duration_seconds'].sum()
            print(f"   Total Audio Duration: {total_audio / 3600:.1f} hours")
        
        if 'transcript_length_words' in df.columns:
            total_words = df['transcript_length_words'].sum()
            print(f"   Total Words Transcribed: {total_words:,}")
    else:
        print("\n⚠️  No database file found")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production SoundCloud ASR batch processor with crash recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full 2 days for test purposes 
    python data_collection.py process --start 2020-01-01 --end 2020-01-31 \
    --output ../data/02_intermediate/transcripts/mustafaa4a_ASR-Somali

  # Process full 5.5 year range in monthly batches
  python data_collection.py process --start 2020-01-01 --end 2025-09-30 \\
      --output ./data/02_intermediate/transcripts/mustafaa4a_ASR-Somali

  # Check current progress
  python data_collection.py progress \\
      --output ./data/02_intermediate/transcripts/mustafaa4a_ASR-Somali
  
  # Resume interrupted collection (same command, auto-skips processed URLs)
  python data_collection.py process --start 2020-01-01 --end 2025-09-30 \\
      --output ./data/02_intermediate/transcripts/mustafaa4a_ASR-Somali
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process date range with batching')
    process_parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    process_parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    process_parser.add_argument('--profile', default='https://soundcloud.com/radio-ergo', 
                               help='SoundCloud profile URL (default: radio-ergo)')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--batch-days', type=int, default=30, 
                               help='Batch size in days (default: 30)')
    process_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Progress command
    progress_parser = subparsers.add_parser('progress', help='Check collection progress')
    progress_parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        run_batch_collection(
            profile_url=args.profile,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            batch_size_days=args.batch_days,
            verbose=args.verbose
        )
    elif args.command == 'progress':
        check_progress(args.output)
    else:
        # Default behavior: run with hardcoded paths
        OUTPUT_DIR = "/teamspace/studios/this_studio/somali-radios-with-ai-for-food-security/data/02_intermediate/transcripts/mustafaa4a_ASR-Somali"
        
        print("💡 Running in default mode. For better control, use:")
        print("   python data_collection.py process --start 2020-01-01 --end 2025-09-30 --output <path>")
        print("   python data_collection.py progress --output <path>\n")
        
        run_batch_collection(
            profile_url="https://soundcloud.com/radio-ergo",
            start_date="2020-01-01",
            end_date="2025-09-30",
            output_dir=OUTPUT_DIR,
            batch_size_days=30,
            verbose=False
        )