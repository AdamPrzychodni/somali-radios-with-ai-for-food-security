"""
SoundCloud Downloader for Linux - Fixed FFmpeg Integration
Properly handles ffmpeg path detection and passing to yt-dlp
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
import yt_dlp
import requests
import time
import hashlib
import json
import shutil
from pathlib import Path


class SoundCloudDownloader:
    def __init__(self, output_dir="downloads"):
        # Convert to absolute path and create directory structure
        self.output_dir = os.path.abspath(output_dir)
        self.ffmpeg_path = None
        self.download_log = {}
        self.log_file = os.path.join(self.output_dir, ".download_log.json")
        
        # Create output directory if it doesn't exist (including parent directories)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory set to: {self.output_dir}")
        
        # Load download log
        self.load_download_log()
        
        # Setup ffmpeg on initialization
        self.setup_ffmpeg()
    
    def load_download_log(self):
        """Load the download log from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.download_log = json.load(f)
                print(f"📋 Loaded download log with {len(self.download_log)} entries")
            except Exception as e:
                print(f"⚠ Could not load download log: {e}")
                self.download_log = {}
        else:
            print("📋 Starting with empty download log")
    
    def save_download_log(self):
        """Save the download log to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.download_log, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save download log: {e}")
    
    def find_ffmpeg(self):
        """Find ffmpeg binary in system."""
        # Check if ffmpeg is in PATH using shutil.which
        ffmpeg_in_path = shutil.which('ffmpeg')
        if ffmpeg_in_path:
            return ffmpeg_in_path
        
        # Common locations to check
        common_locations = [
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',
            '/snap/bin/ffmpeg',
            '~/.local/bin/ffmpeg',
        ]
        
        for location in common_locations:
            expanded_path = os.path.expanduser(location)
            if os.path.exists(expanded_path) and os.access(expanded_path, os.X_OK):
                return expanded_path
        
        return None
    
    def setup_ffmpeg(self):
        """Setup ffmpeg for Linux systems - improved detection."""
        print("🔧 Checking for ffmpeg installation...")
        
        # First try to find existing ffmpeg
        self.ffmpeg_path = self.find_ffmpeg()
        
        if self.ffmpeg_path:
            # Verify it actually works
            if self.verify_ffmpeg(self.ffmpeg_path):
                print(f"✓ Found working ffmpeg at: {self.ffmpeg_path}")
                return True
            else:
                print(f"⚠ Found ffmpeg at {self.ffmpeg_path} but it doesn't work properly")
                self.ffmpeg_path = None
        
        # If not found, try to install it
        print("ffmpeg not found. Attempting to install...")
        
        if self.install_ffmpeg():
            # Re-check after installation
            self.ffmpeg_path = self.find_ffmpeg()
            if self.ffmpeg_path and self.verify_ffmpeg(self.ffmpeg_path):
                print(f"✓ Successfully installed ffmpeg at: {self.ffmpeg_path}")
                return True
        
        print("⚠ Failed to install ffmpeg automatically.")
        print("Please install it manually:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  Fedora: sudo dnf install ffmpeg")
        print("  Arch: sudo pacman -S ffmpeg")
        return False
    
    def verify_ffmpeg(self, ffmpeg_path):
        """Verify that ffmpeg actually works."""
        try:
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and 'ffmpeg version' in result.stdout
        except Exception as e:
            print(f"Error verifying ffmpeg: {e}")
            return False
    
    def install_ffmpeg(self):
        """Try to install ffmpeg based on detected OS."""
        try:
            # Detect the Linux distribution
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    os_info = f.read().lower()
                
                if 'ubuntu' in os_info or 'debian' in os_info:
                    print("Installing ffmpeg using apt...")
                    subprocess.run(['sudo', 'apt-get', 'update'], check=False, capture_output=True)
                    result = subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], 
                                          capture_output=True, text=True)
                    return result.returncode == 0
                    
                elif 'fedora' in os_info:
                    print("Installing ffmpeg using dnf...")
                    result = subprocess.run(['sudo', 'dnf', 'install', '-y', 'ffmpeg'], 
                                          capture_output=True, text=True)
                    return result.returncode == 0
                    
                elif 'arch' in os_info:
                    print("Installing ffmpeg using pacman...")
                    result = subprocess.run(['sudo', 'pacman', '-S', '--noconfirm', 'ffmpeg'], 
                                          capture_output=True, text=True)
                    return result.returncode == 0
        except Exception as e:
            print(f"Error during installation: {e}")
        
        return False
    
    def validate_url(self, url):
        """Validate if the URL is a valid SoundCloud URL."""
        pattern = r'^https?://(?:www\.)?soundcloud\.com/[\w-]+/[\w-]+'
        return bool(re.match(pattern, url))
    
    def get_file_hash(self, url):
        """Generate a unique hash for a URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def is_already_downloaded(self, url):
        """Check if a URL has already been successfully downloaded."""
        url_hash = self.get_file_hash(url)
        
        if url_hash in self.download_log:
            file_path = self.download_log[url_hash].get('file_path')
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 1000000:  # > 1MB
                    print(f"✓ Already downloaded: {os.path.basename(file_path)}")
                    return True
        
        return False
    
    def sanitize_filename(self, filename):
        """Sanitize filename to remove problematic characters."""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', ' ', filename)  # Replace multiple spaces with single space
        filename = filename.strip()
        
        # Ensure filename isn't too long (limit to 200 chars)
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    def download_audio(self, url, max_retries=3):
        """Download audio from SoundCloud URL with proper ffmpeg configuration."""
        if not self.validate_url(url):
            print(f"✗ Invalid URL: {url}")
            return None
        
        # Check if already downloaded
        if self.is_already_downloaded(url):
            return self.download_log[self.get_file_hash(url)]['file_path']
        
        print(f"⬇ Downloading: {url}")
        
        # Configure yt-dlp options with sanitized filename template
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'restrictfilenames': True,  # Use only ASCII characters in filenames
            # Audio extraction settings
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'prefer_ffmpeg': True,  # Prefer ffmpeg over avconv
        }
        
        # CRITICAL: Set ffmpeg location properly
        if self.ffmpeg_path:
            # Get the directory containing ffmpeg
            ffmpeg_dir = os.path.dirname(self.ffmpeg_path)
            ydl_opts['ffmpeg_location'] = ffmpeg_dir
            print(f"  Using ffmpeg from: {ffmpeg_dir}")
        
        # Try downloading with retries
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract info first
                    info = ydl.extract_info(url, download=False)
                    
                    # Generate expected filename
                    filename = ydl.prepare_filename(info)
                    base, _ = os.path.splitext(filename)
                    mp3_file = f"{base}.mp3"
                    
                    # Ensure the mp3_file path is within our output directory
                    mp3_file = os.path.join(self.output_dir, os.path.basename(mp3_file))
                    
                    # Check if file already exists
                    if os.path.exists(mp3_file) and os.path.getsize(mp3_file) > 1000000:
                        print(f"✓ File already exists: {os.path.basename(mp3_file)}")
                        # Add to log
                        self.download_log[self.get_file_hash(url)] = {
                            'url': url,
                            'file_path': mp3_file,
                            'download_date': datetime.now().isoformat()
                        }
                        self.save_download_log()
                        return mp3_file
                    
                    # Now download
                    ydl.download([url])
                    
                    # Find the actual downloaded file (yt-dlp might change the filename)
                    downloaded_files = []
                    for file in os.listdir(self.output_dir):
                        if file.endswith('.mp3'):
                            file_path = os.path.join(self.output_dir, file)
                            if os.path.getmtime(file_path) > time.time() - 300:  # Modified in last 5 minutes
                                downloaded_files.append(file_path)
                    
                    # Use the most recently modified file
                    if downloaded_files:
                        mp3_file = max(downloaded_files, key=os.path.getmtime)
                    
                    # Verify the download
                    if os.path.exists(mp3_file) and os.path.getsize(mp3_file) > 0:
                        print(f"✓ Successfully downloaded: {os.path.basename(mp3_file)}")
                        print(f"  📁 Saved to: {mp3_file}")
                        
                        # Add to download log
                        self.download_log[self.get_file_hash(url)] = {
                            'url': url,
                            'file_path': mp3_file,
                            'download_date': datetime.now().isoformat()
                        }
                        self.save_download_log()
                        
                        return mp3_file
                    else:
                        raise Exception("Downloaded file is empty or missing")
                        
            except Exception as e:
                error_msg = str(e)[:100]
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                
                # If ffmpeg issue, try to fix it
                if 'ffmpeg' in error_msg.lower() and attempt == 0:
                    print("  Attempting to fix ffmpeg configuration...")
                    self.setup_ffmpeg()
                
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
        
        print(f"✗ Failed to download after {max_retries} attempts")
        return None
    
    def download_date_range(self, profile_url, start_date, end_date):
        """
        Download tracks from a SoundCloud profile within a date range.
        
        Args:
            profile_url: SoundCloud profile URL
            start_date: Start date (datetime object or string 'YYYY-MM-DD')
            end_date: End date (datetime object or string 'YYYY-MM-DD')
        
        Returns:
            Tuple of (successful_downloads, failed_urls)
        """
        # Parse dates if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"\n📅 Searching for tracks from {start_date.date()} to {end_date.date()}")
        print(f"📁 Output directory: {self.output_dir}\n")
        
        # Generate URLs for date range
        urls = self.generate_urls_for_range(profile_url, start_date, end_date)
        
        if not urls:
            print("No URLs found for the specified date range")
            return [], []
        
        print(f"Found {len(urls)} potential tracks to download\n")
        
        successful = []
        failed = []
        
        for i, (date, url) in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Processing {date.date()}")
            
            # Check if URL exists (optional - can be skipped for faster processing)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code == 404:
                    print(f"  ✗ URL not found (404)")
                    failed.append(url)
                    continue
            except:
                pass  # Try downloading anyway
            
            result = self.download_audio(url)
            
            if result:
                successful.append(result)
            else:
                failed.append(url)
            
            # Small delay between downloads
            if i < len(urls):
                time.sleep(2)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Download Summary:")
        print(f"  ✓ Successful: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        print(f"  📁 Files saved to: {self.output_dir}")
        
        # List downloaded files
        if successful:
            print(f"\nDownloaded files:")
            for file_path in successful:
                print(f"  • {os.path.basename(file_path)}")
        
        print(f"{'='*50}\n")
        
        return successful, failed
    
    def generate_urls_for_range(self, profile_url, start_date, end_date):
        """Generate potential URLs for a date range."""
        urls = []
        
        # Clean profile URL
        profile_url = profile_url.rstrip('/')
        
        current_date = start_date
        while current_date <= end_date:
            day = current_date.day
            month = current_date.strftime('%b').lower()
            year = current_date.year
            
            # Generate potential URL formats (most common first)
            potential_urls = [
                f"{profile_url}/idaacadda-{day:02d}-{month}-{year}",
                f"{profile_url}/idaacadda-{day}-{month}-{year}",
            ]
            
            # Add the first URL for this date
            urls.append((current_date, potential_urls[0]))
            
            current_date += timedelta(days=1)
        
        return urls


# Usage example with explicit path handling
def main():
    """Main function for command-line usage."""
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"🏠 Current directory: {current_dir}")
    
    # Set the output directory relative to current location
    output_dir = "data/01_raw"
    
    # If running from project root, this should work
    # If running from elsewhere, construct the full path
    if os.path.basename(current_dir) == "somali-radios-with-ai-for-food-security":
        full_output_path = os.path.join(current_dir, output_dir)
    else:
        # Try to find the project directory
        project_name = "somali-radios-with-ai-for-food-security"
        if project_name in current_dir:
            # We're inside the project somewhere
            project_root = current_dir.split(project_name)[0] + project_name
            full_output_path = os.path.join(project_root, output_dir)
        else:
            # Default to creating the structure in current directory
            full_output_path = os.path.join(current_dir, project_name, output_dir)
    
    print(f"🎯 Target output directory: {full_output_path}")
    
    downloader = SoundCloudDownloader(output_dir=full_output_path)
    
    # Download a specific date range
    profile_url = "https://soundcloud.com/radio-ergo"
    start_date = "2024-07-01"
    end_date = "2024-07-01"
    
    successful, failed = downloader.download_date_range(
        profile_url,
        start_date,
        end_date
    )
    
    # Retry failed downloads if any
    if failed:
        print("\n🔄 Retrying failed downloads...")
        retry_successful = []
        still_failed = []
        
        for url in failed:
            result = downloader.download_audio(url, max_retries=2)
            if result:
                retry_successful.append(result)
            else:
                still_failed.append(url)
        
        if retry_successful:
            print(f"✓ Successfully downloaded {len(retry_successful)} on retry")
        if still_failed:
            print(f"✗ Still failed: {len(still_failed)} URLs")
            for url in still_failed:
                print(f"  - {url}")


# Alternative function for direct usage when you know your exact path
def download_to_specific_path():
    """Function to download directly to your project's data directory."""
    # Hardcoded path to your project structure
    project_root = "somali-radios-with-ai-for-food-security"
    output_path = os.path.join(project_root, "data", "01_raw")
    
    # Create the full path if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"📁 Downloading to: {os.path.abspath(output_path)}")
    
    downloader = SoundCloudDownloader(output_dir=output_path)
    
    # Your download parameters
    profile_url = "https://soundcloud.com/radio-ergo"
    start_date = "2024-07-01"
    end_date = "2024-07-01"
    
    return downloader.download_date_range(profile_url, start_date, end_date)