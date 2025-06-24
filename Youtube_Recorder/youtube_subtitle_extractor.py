import yt_dlp
import threading
import time
import re
import os
from datetime import datetime, timedelta

class RealtimeSubtitleExtractor:
    def __init__(self, output_dir="subtitles"):
        self.output_dir = output_dir
        self.running = False
        self.subtitle_thread = None
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_live_subtitles(self, url, languages=['en'], display_live=True, 
                             save_to_file=True, update_interval=2):
        """
        Extract and display live subtitles from YouTube stream
        
        Args:
            url: YouTube live stream URL
            languages: List of language codes to extract
            display_live: Whether to print subtitles to console in real-time
            save_to_file: Whether to save subtitles to file
            update_interval: How often to check for new subtitles (seconds)
        """
        
        self.running = True
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Set up file logging if requested
        subtitle_files = {}
        if save_to_file:
            for lang in languages:
                filename = os.path.join(self.output_dir, f"live_subs_{lang}_{timestamp}.txt")
                subtitle_files[lang] = open(filename, 'w', encoding='utf-8')
        
        try:
            # Get initial subtitle information
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check available subtitles
                auto_captions = info.get('automatic_captions', {})
                manual_subs = info.get('subtitles', {})
                
                print(f"Available auto captions: {list(auto_captions.keys())}")
                print(f"Available manual subtitles: {list(manual_subs.keys())}")
                
                for lang in languages:
                    if lang in auto_captions or lang in manual_subs:
                        print(f"Starting subtitle extraction for: {lang}")
                        
                        # Start subtitle extraction thread for each language
                        thread = threading.Thread(
                            target=self._extract_subtitle_stream,
                            args=(url, lang, display_live, subtitle_files.get(lang), update_interval)
                        )
                        thread.daemon = True
                        thread.start()
                    else:
                        print(f"Language '{lang}' not available for this stream")
                
                # Keep main thread alive
                while self.running:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nStopping subtitle extraction...")
            self.running = False
        except Exception as e:
            print(f"Error during subtitle extraction: {e}")
        finally:
            # Close files
            for file_handle in subtitle_files.values():
                if file_handle:
                    file_handle.close()
    
    def _extract_subtitle_stream(self, url, language, display_live, file_handle, update_interval):
        """Extract subtitles for a specific language"""
        
        seen_subtitles = set()
        
        while self.running:
            try:
                # Download current subtitle chunk
                temp_filename = f"temp_sub_{language}_{int(time.time())}"
                
                ydl_opts = {
                    'outtmpl': temp_filename + '.%(ext)s',
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': [language],
                    'subtitlesformat': 'vtt',
                    'skip_download': True,
                    'quiet': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find and read the subtitle file
                subtitle_file = None
                for file in os.listdir('.'):
                    if file.startswith(temp_filename) and file.endswith('.vtt'):
                        subtitle_file = file
                        break
                
                if subtitle_file and os.path.exists(subtitle_file):
                    new_subs = self._parse_vtt_file(subtitle_file)
                    
                    # Process new subtitles
                    for timestamp, text in new_subs:
                        if text not in seen_subtitles:
                            seen_subtitles.add(text)
                            
                            if display_live:
                                print(f"[{language}] {timestamp}: {text}")
                            
                            if file_handle:
                                file_handle.write(f"{timestamp}: {text}\n")
                                file_handle.flush()
                    
                    # Clean up temp file
                    os.remove(subtitle_file)
                
                time.sleep(update_interval)
                
            except Exception as e:
                if self.running:  # Only show error if we're still supposed to be running
                    print(f"Error extracting subtitles for {language}: {e}")
                time.sleep(update_interval)
    
    def _parse_vtt_file(self, filename):
        """Parse VTT subtitle file and extract text with timestamps"""
        subtitles = []
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newlines to get subtitle blocks
            blocks = content.split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 2:
                    # Look for timestamp line (contains -->)
                    timestamp_line = None
                    text_lines = []
                    
                    for line in lines:
                        if '-->' in line:
                            timestamp_line = line
                        elif line.strip() and not line.startswith('WEBVTT'):
                            # Clean up subtitle text
                            clean_text = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
                            if clean_text.strip():
                                text_lines.append(clean_text.strip())
                    
                    if timestamp_line and text_lines:
                        # Extract start time
                        start_time = timestamp_line.split(' --> ')[0]
                        subtitle_text = ' '.join(text_lines)
                        subtitles.append((start_time, subtitle_text))
        
        except Exception as e:
            print(f"Error parsing VTT file: {e}")
        
        return subtitles
    
    def stop_extraction(self):
        """Stop subtitle extraction"""
        self.running = False

class SubtitleDisplay:
    """Class for displaying subtitles in various formats"""
    
    @staticmethod
    def display_with_formatting(timestamp, text, language='en'):
        """Display subtitle with nice formatting"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"Time: {current_time} | Stream: {timestamp} | Lang: {language.upper()}")
        print(f"{'='*60}")
        print(f"{text}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def save_as_srt(subtitles, filename):
        """Save subtitles in SRT format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (timestamp, text) in enumerate(subtitles, 1):
                # Convert timestamp format if needed
                f.write(f"{i}\n")
                f.write(f"{timestamp} --> {timestamp}\n")  # Simplified
                f.write(f"{text}\n\n")

# Usage example with live display
class LiveSubtitleMonitor:
    def __init__(self, output_dir="subtitles"):
        self.extractor = RealtimeSubtitleExtractor(output_dir=output_dir)
        self.display = SubtitleDisplay()
    
    def monitor_stream(self, url, languages=['en'], show_formatted=True):
        """Monitor stream and display subtitles with formatting"""
        
        print(f"Starting live subtitle monitoring for: {url}")
        print(f"Languages: {languages}")
        print("Press Ctrl+C to stop\n")
        
        try:
            if show_formatted:
                # Custom extraction with formatted display
                self._custom_formatted_extraction(url, languages)
            else:
                # Standard extraction
                self.extractor.extract_live_subtitles(
                    url, 
                    languages=languages,
                    display_live=True,
                    save_to_file=True
                )
        except KeyboardInterrupt:
            print("\nStopping subtitle monitoring...")
            self.extractor.stop_extraction()
    
    def _custom_formatted_extraction(self, url, languages):
        """Custom extraction with better formatting"""
        
        def subtitle_callback(timestamp, text, lang):
            self.display.display_with_formatting(timestamp, text, lang)
        
        # This would integrate with the extraction logic
        # Implementation would depend on specific formatting needs
        self.extractor.extract_live_subtitles(
            url, 
            languages=languages,
            display_live=False,  # We'll handle display ourselves
            save_to_file=True
        )

# Example usage
if __name__ == "__main__":
    
    stream_url = "https://www.youtube.com/watch?v=0_DjDdfqtUE"
    # stream_url = "https://www.youtube.com/watch?v=51iONeETSng"

    # Basic subtitle extraction
    extractor = RealtimeSubtitleExtractor(output_dir="subtitle")
    # Extract subtitles for multiple languages
    print("Starting real-time subtitle extraction...")
    extractor.extract_live_subtitles(
        stream_url,
        languages=['en-JkeT_87f4cc', 'zh-Hant'],  # English and Spanish
        display_live=True,
        save_to_file=True,
        update_interval=2
    )

    # Or use the monitor for better formatting
    # monitor = LiveSubtitleMonitor(output_dir="subtitle")
    # monitor.monitor_stream(stream_url, languages=['en'])
