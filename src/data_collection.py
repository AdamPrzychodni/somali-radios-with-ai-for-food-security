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
from yt_dlp.utils import DownloadError 
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
from tqdm import tqdm

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ============================================================================
# GPU-OPTIMIZED ASR ENGINE
# ============================================================================

class SomaliASREngine:
    """
    GPU-optimized transcription engine with batched inference.
    Utilizes L4 Tensor Cores for 3-5x speedup over CPU.
    """
    
    def __init__(
        self, 
        model_name: str = "Mustafaa4a/ASR-Somali",
        batch_size: int = 8,
        use_fp16: bool = True,
        verbose: bool = False
    ):
        """
        Initialize GPU-optimized ASR model.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of audio chunks to process simultaneously
            use_fp16: Enable mixed precision for faster inference on L4
            verbose: Enable detailed logging
        """
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2ForCTC] = None
        self.device: str = "cpu"
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.verbose = verbose
        self._setup_model(model_name)

    def _setup_model(self, model_name: str) -> None:
        """
        Load model with GPU optimizations.
        
        Raises:
            RuntimeError: If model fails to load
        """
        if self.verbose:
            print(f"Loading ASR model: {model_name}")
            
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            # Force GPU usage
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.device == "cuda":
                self.model.to(self.device)
                
                # Enable mixed precision on L4 (Tensor Cores)
                if self.use_fp16:
                    self.model = self.model.half()
                    if self.verbose:
                        print("✓ FP16 mixed precision enabled")
                
                # Optimize for inference
                self.model.eval()
                torch.backends.cudnn.benchmark = True
                
                if self.verbose:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✓ Model loaded on GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
                    print(f"✓ Batch size: {self.batch_size}")
            else:
                self.model.to(self.device)
                if self.verbose:
                    print("⚠️  Running on CPU (slower)")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model: {e}")

    def transcribe_from_memory_batched(self, audio_data: bytes) -> str:
        """
        GPU-accelerated batched transcription.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
            
        Raises:
            ValueError: If transcription fails
            RuntimeError: If model not initialized
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("ASR model is not initialized.")
            
        target_sr = 16000
        
        try:
            # Load audio
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=target_sr)
            
            # Larger chunks for GPU batch processing (60s chunks for L4)
            chunk_length_s = 60
            chunk_length = chunk_length_s * target_sr
            
            # Create overlapping chunks for better accuracy
            overlap = int(chunk_length * 0.1)  # 10% overlap
            chunks = []
            
            for i in range(0, len(audio), chunk_length - overlap):
                chunk = audio[i:i + chunk_length]
                if len(chunk) > target_sr:  # Skip chunks < 1 second
                    chunks.append(chunk)
            
            if not chunks:
                return ""
            
            # Process chunks in batches
            transcriptions = []
            
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[batch_start:batch_start + self.batch_size]
                
                # Prepare batch with padding
                inputs = self.processor(
                    batch_chunks,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True
                )
                
                input_values = inputs.input_values.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                # Mixed precision inference
                with torch.no_grad():
                    if self.use_fp16 and self.device == "cuda":
                        input_values = input_values.half()
                    
                    logits = self.model(
                        input_values,
                        attention_mask=attention_mask
                    ).logits
                
                # Decode batch
                predicted_ids = torch.argmax(logits, dim=-1)
                batch_transcriptions = self.processor.batch_decode(predicted_ids)
                transcriptions.extend(batch_transcriptions)
            
            # Join with space, remove excessive whitespace
            full_text = " ".join(transcriptions)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return full_text

        except Exception as e:
            raise ValueError(f"Transcription failed: {e}")

    def get_vram_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with allocated and reserved VRAM in GB
        """
        if self.device == "cuda":
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            }
        return {'allocated_gb': 0, 'reserved_gb': 0}


# ============================================================================
# OPTIMIZED DOWNLOADER & PROCESSOR
# ============================================================================

class SilentLogger:
    """Silent logger for yt-dlp."""
    def debug(self, msg: str) -> None:
        pass
    def warning(self, msg: str) -> None:
        pass
    def error(self, msg: str) -> None:
        pass


class StreamingSoundCloudDownloader:
    """
    GPU-optimized batch processor for SoundCloud audio.
    """
    
    def __init__(
        self, 
        output_dir: str,
        output_format: str = "structured",
        batch_size: int = 8,
        use_fp16: bool = True,
        verbose: bool = False
    ):
        """
        Initialize downloader with GPU optimizations.
        
        Args:
            output_dir: Directory for outputs
            output_format: 'structured' (CSV), 'txt', or 'both'
            batch_size: GPU batch size for inference
            use_fp16: Enable FP16 mixed precision
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
        
        # Initialize GPU-optimized engine
        self.transcription_engine = SomaliASREngine(
            batch_size=batch_size,
            use_fp16=use_fp16,
            verbose=verbose
        )
        
        if verbose:
            vram = self.transcription_engine.get_vram_usage()
            print(f"Initial VRAM: {vram['allocated_gb']:.2f} GB allocated, "
                  f"{vram['reserved_gb']:.2f} GB reserved")

    def _find_ffmpeg(self) -> Optional[str]:
        """Locate ffmpeg binary."""
        if self.verbose:
            print("Checking for ffmpeg...")
            
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            if self.verbose:
                print(f"✓ Found ffmpeg: {ffmpeg_path}")
            return ffmpeg_path
        
        for location in ['/usr/bin/ffmpeg', '/home/zeus/miniconda3/bin/ffmpeg']:
            if Path(location).exists():
                if self.verbose:
                    print(f"✓ Found ffmpeg: {location}")
                return location
                
        if self.verbose:
            print("⚠️ ffmpeg not found")
        return None

    def _init_structured_storage(self) -> None:
        """Initialize CSV database."""
        if not self.structured_data_file.exists():
            columns = [
                'id', 'url', 'title', 'date_recorded', 'date_processed',
                'processing_duration_seconds', 'audio_size_mb', 'audio_duration_seconds',
                'transcript_length_chars', 'transcript_length_words', 'transcript_text'
            ]
            pd.DataFrame(columns=columns).to_csv(self.structured_data_file, index=False)

    def _load_transcription_log(self) -> Dict:
        """Load processing log."""
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
        """Download audio to memory buffer."""
        metadata: Dict = {'success': False, 'error': None, 'info': {}}
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'logger': SilentLogger(),
        }
        
        if self.ffmpeg_path:
            ydl_opts['ffmpeg_location'] = self.ffmpeg_path

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                info = info or {}
                metadata['info'] = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                }
            
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
                    
        except DownloadError as e:
            if 'HTTP Error 404' in str(e):
                metadata['error'] = 'Broadcast not found (404 Error).'
            else:
                metadata['error'] = f'Download failed: {str(e)}'
            return None, metadata
        except Exception as e:
            metadata['error'] = str(e)
            return None, metadata
            
        return None, metadata

    def _transcribe_audio_data(self, audio_data: bytes, metadata: Dict) -> Tuple[Optional[str], Dict]:
        """GPU-accelerated transcription."""
        start_time = time.time()
        
        if not audio_data:
            metadata['transcription_error'] = "Empty audio data"
            return None, metadata
            
        try:
            metadata['audio_size_mb'] = len(audio_data) / (1024 * 1024)
            
            if PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                metadata['audio_duration_seconds'] = len(audio_segment) / 1000.0

            # Use batched GPU inference
            transcript = self.transcription_engine.transcribe_from_memory_batched(audio_data)
            
            # Track VRAM usage
            vram = self.transcription_engine.get_vram_usage()
            
            metadata.update({
                'processing_duration_seconds': time.time() - start_time,
                'transcript_length_chars': len(transcript),
                'transcript_length_words': len(transcript.split()),
                'transcription_success': True,
                'vram_used_gb': vram['allocated_gb'],
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
        """Save transcript as text file."""
        transcript_file = self.output_dir / f"{base_filename}.txt"
        
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"# Transcription Report\n{'='*50}\n")
            f.write(f"Source URL: {metadata.get('url', 'Unknown')}\n")
            f.write(f"Title: {metadata.get('info', {}).get('title', 'Unknown')}\n")
            f.write(f"Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Mustafaa4a/ASR-Somali (GPU-optimized)\n")
            f.write(f"Audio Duration: {metadata.get('audio_duration_seconds', 0):.1f}s\n")
            f.write(f"Processing Time: {metadata.get('processing_duration_seconds', 0):.1f}s\n")
            f.write(f"VRAM Used: {metadata.get('vram_used_gb', 0):.2f} GB\n")
            f.write(f"Word Count: {metadata.get('transcript_length_words', 0)}\n")
            f.write(f"{'='*50}\n\n## Transcript\n\n{transcript}")
            
        return str(transcript_file)

    def _process_url(self, url: str) -> Tuple[bool, Optional[str], Dict]:
        """Process single URL with GPU acceleration."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        if url_hash in self.transcription_log:
            return True, self.transcription_log[url_hash].get('file_path'), self.transcription_log[url_hash]
        
        audio_data, metadata = self._download_to_memory(url)
        if not audio_data:
            return False, None, metadata
            
        metadata['url'] = url
        metadata['id'] = url_hash
        
        transcript, metadata = self._transcribe_audio_data(audio_data, metadata)
        if not transcript:
            return False, None, metadata
        
        safe_title = re.sub(r'[^\w\s-]', '', metadata.get('info', {}).get('title', ''))
        base_filename = re.sub(r'[-\s]+', '-', safe_title).strip('-') or f"soundcloud_{url_hash[:8]}"

        file_path: Optional[str] = None
        
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
        
        if self.output_format in ['txt', 'both']:
            txt_file = self._save_transcript_file(transcript, metadata, base_filename)
            file_path = txt_file
        
        self.transcription_log[url_hash] = {
            'url': url,
            'title': metadata.get('info', {}).get('title', 'Unknown'),
            'file_path': file_path,
            'success': True
        }
        self._save_transcription_log()
        
        return True, file_path, metadata

    def _generate_urls_for_range(self, profile_url: str, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, str]]:
        """Generate SoundCloud URLs for date range."""
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
        """Process date range with GPU acceleration."""
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD'")

        print(f"\n{'='*70}")
        print(f"GPU-Optimized SoundCloud ASR Pipeline")
        print(f"{'='*70}")
        print(f"Profile: {profile_url}")
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Device: {self.transcription_engine.device.upper()}")
        print(f"Batch Size: {self.transcription_engine.batch_size}")
        print(f"Output: {self.structured_data_file}")
        print(f"{'='*70}\n")
        
        urls = self._generate_urls_for_range(profile_url, start_date, end_date)
        
        if not urls:
            print("No URLs generated for date range")
            return {'successful': [], 'failed': [], 'skipped': []}
        
        results: Dict[str, List] = {'successful': [], 'failed': [], 'skipped': []}
        
        with tqdm(urls, desc="Processing tracks", unit="track") as pbar:
            for date, url in pbar:
                pbar.set_postfix_str(f"{date.date()}")
                
                try:
                    success, _, metadata = self._process_url(url)
                    if success:
                        results['successful'].append(url)
                        # Show speedup in progress bar
                        if 'processing_duration_seconds' in metadata and 'audio_duration_seconds' in metadata:
                            speedup = metadata['audio_duration_seconds'] / max(metadata['processing_duration_seconds'], 0.1)
                            pbar.set_postfix_str(f"{date.date()} | {speedup:.1f}x realtime")
                    else:
                        results['failed'].append(url)
                        error_msg = metadata.get('error')
                        if error_msg and 'Broadcast not found' in error_msg:
                            tqdm.write(f"↪️  Skipping {date.date()}: Broadcast not found.")
                        elif self.verbose and error_msg:
                            tqdm.write(f"✗ Error on {date.date()}: {error_msg}")

                except Exception as e:
                    if self.verbose:
                        tqdm.write(f"✗ Error on {date.date()}: {e}")
                    results['failed'].append(url)
                
                time.sleep(0.3)  # Reduced wait time due to faster processing

        print(f"\n{'='*70}")
        print("Processing Complete")
        print(f"{'='*70}")
        print(f"✓ Successful: {len(results['successful'])}")
        print(f"✗ Failed: {len(results['failed'])}")
        
        # Show final VRAM usage
        vram = self.transcription_engine.get_vram_usage()
        print(f"Final VRAM: {vram['allocated_gb']:.2f} GB allocated")
        print(f"Database: {self.structured_data_file}")
        print(f"{'='*70}\n")
        
        return results


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def run_batch_collection(
    profile_url: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    batch_size_days: int = 30,
    gpu_batch_size: int = 8,
    use_fp16: bool = True,
    verbose: bool = False
) -> None:
    """
    GPU-optimized batch collection with crash recovery.
    
    Args:
        profile_url: SoundCloud profile URL
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        output_dir: Output directory path
        batch_size_days: Process in chunks of N days
        gpu_batch_size: GPU inference batch size (higher = more VRAM, faster)
        use_fp16: Enable FP16 mixed precision (recommended for L4)
        verbose: Enable detailed logging
    """
    print(f"\n{'='*80}")
    print("GPU-OPTIMIZED PRODUCTION DATA COLLECTION")
    print(f"{'='*80}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Batch Size: {batch_size_days} days")
    print(f"GPU Batch Size: {gpu_batch_size}")
    print(f"FP16 Precision: {use_fp16}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1
    
    # Estimate speedup (3-5x faster on GPU)
    estimated_minutes = total_days * 3 / 4  # ~45 seconds per track on GPU
    
    print(f"📊 Total Scope: {total_days} days")
    print(f"📦 Estimated Batches: {(total_days + batch_size_days - 1) // batch_size_days}")
    print(f"⏱️  Estimated Time: ~{estimated_minutes / 60:.1f} hours (GPU-accelerated)\n")
    
    current_start = start
    batch_num = 1
    total_successful = 0
    total_failed = 0
    
    while current_start <= end:
        batch_end = min(
            current_start + timedelta(days=batch_size_days - 1),
            end
        )
        
        print(f"\n{'─'*80}")
        print(f"🔄 BATCH {batch_num}: {current_start.date()} → {batch_end.date()}")
        print(f"{'─'*80}")
        
        try:
            downloader = StreamingSoundCloudDownloader(
                output_dir=output_dir,
                output_format="structured",
                batch_size=gpu_batch_size,
                use_fp16=use_fp16,
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
            break
            
        except Exception as e:
            print(f"\n❌ ERROR in batch {batch_num}: {e}")
            print(f"📍 Progress saved. Continuing to next batch...")
        
        current_start = batch_end + timedelta(days=1)
        batch_num += 1
    
    print(f"\n\n{'='*80}")
    print("📊 COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Successful: {total_successful}")
    print(f"Total Failed: {total_failed}")
    print(f"Database: {Path(output_dir) / 'transcriptions_database.csv'}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-optimized SoundCloud ASR")
    
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--profile', default='https://soundcloud.com/radio-ergo')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--batch-days', type=int, default=30)
    parser.add_argument('--gpu-batch-size', type=int, default=8, 
                       help='GPU inference batch size (4-16 recommended)')
    parser.add_argument('--no-fp16', action='store_true', 
                       help='Disable FP16 mixed precision')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    run_batch_collection(
        profile_url=args.profile,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        batch_size_days=args.batch_days,
        gpu_batch_size=args.gpu_batch_size,
        use_fp16=not args.no_fp16,
        verbose=args.verbose
    )
    