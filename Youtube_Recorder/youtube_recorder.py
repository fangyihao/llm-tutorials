import yt_dlp
import subprocess
import threading
import os
import time
import psutil

class RealtimeCompressRecorder:
    def __init__(self, output_dir="compressed_recordings"):
        self.output_dir = output_dir
        self.recording = False
        os.makedirs(output_dir, exist_ok=True)
    
    def record_with_realtime_compression(self, url, duration_minutes=None, 
                                       target_filesize_mb=None, quality='best'):
        """
        Record and compress in real-time using external downloader
        
        Args:
            url: YouTube live stream URL
            duration_minutes: Recording duration
            target_filesize_mb: Target file size in MB (optional)
            quality: Stream quality
        """
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"compressed_{timestamp}.mp4")
        
        # Calculate bitrate if target filesize is specified
        video_bitrate = None
        if target_filesize_mb and duration_minutes:
            # Rough calculation: target_size_mb * 8 / duration_minutes / 60 * 0.8 (80% for video)
            video_bitrate = int(target_filesize_mb * 8 * 1024 / (duration_minutes * 60) * 0.8)
            print(f"Target filesize: {target_filesize_mb}MB, calculated video bitrate: {video_bitrate}k")
        
        # Build ffmpeg command for real-time compression
        ffmpeg_args = self._build_compression_command(video_bitrate, duration_minutes)
        
        ydl_opts = {
            'outtmpl': output_file,
            'format': quality,
            'live_from_start': True,
            'wait_for_video': (1, 30),
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg': ffmpeg_args
            }
        }
        
        try:
            print(f"Starting real-time compressed recording...")
            print(f"Output: {output_file}")
            
            self.recording = True
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            print("Recording completed successfully")
            
            # Show file size
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"Final file size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            self.recording = False
    
    def _build_compression_command(self, video_bitrate=None, duration_minutes=None):
        """Build ffmpeg compression command"""
        
        cmd = [
            '-c:v', 'libx264',
            '-preset', 'faster',  # Faster preset for real-time
            '-tune', 'zerolatency',  # Optimize for low latency
        ]
        
        if video_bitrate:
            cmd.extend([
                '-b:v', f'{video_bitrate}k',
                '-maxrate', f'{int(video_bitrate * 1.2)}k',
                '-bufsize', f'{int(video_bitrate * 2)}k'
            ])
        else:
            cmd.extend(['-crf', '23'])  # Default quality
        
        # Audio settings
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '96k',
            '-ar', '44100'
        ])
        
        # Duration limit
        if duration_minutes:
            cmd.extend(['-t', str(duration_minutes * 60)])
        
        # Output optimization
        cmd.extend([
            '-movflags', '+faststart',
            '-f', 'mp4'
        ])
        
        return cmd
    
    def record_adaptive_compression(self, url, duration_minutes=None):
        """
        Record with adaptive compression based on system resources
        """
        
        # Check system resources
        cpu_count = psutil.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        print(f"System info - CPU cores: {cpu_count}, Available RAM: {available_memory:.1f}GB")
        
        # Adjust settings based on system capabilities
        if cpu_count >= 8 and available_memory >= 8:
            preset = 'medium'
            crf = '20'
            print("Using high-quality settings (good system)")
        elif cpu_count >= 4 and available_memory >= 4:
            preset = 'fast'
            crf = '23'
            print("Using balanced settings (average system)")
        else:
            preset = 'faster'
            crf = '26'
            print("Using fast settings (limited system)")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"adaptive_{timestamp}.mp4")
        
        ffmpeg_args = [
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', crf,
            '-c:a', 'aac',
            '-b:a', '96k',
            '-movflags', '+faststart'
        ]
        
        if duration_minutes:
            ffmpeg_args.extend(['-t', str(duration_minutes * 60)])
        
        ydl_opts = {
            'outtmpl': output_file,
            'format': 'best',
            'live_from_start': True,
            'wait_for_video': (1, 30),
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg': ffmpeg_args
            }
        }
        
        try:
            self.recording = True
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.recording = False
    
    def record_ultra_compressed(self, url, duration_minutes=None):
        """
        Record with maximum compression for minimal file size
        """
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"ultra_compressed_{timestamp}.mp4")
        
        # Ultra compression settings
        ffmpeg_args = [
            '-c:v', 'libx265',  # Better compression than x264
            '-preset', 'slow',   # Better compression ratio
            '-crf', '30',        # Higher CRF = more compression
            '-maxrate', '1M',    # Limit bitrate
            '-bufsize', '2M',
            '-c:a', 'aac',
            '-b:a', '64k',       # Lower audio bitrate
            '-ar', '44100',
            '-movflags', '+faststart',
            '-tag:v', 'hvc1'     # HEVC compatibility
        ]
        
        if duration_minutes:
            ffmpeg_args.extend(['-t', str(duration_minutes * 60)])
        
        ydl_opts = {
            'outtmpl': output_file,
            'format': 'best[height<=720]',  # Limit resolution for smaller files
            'live_from_start': True,
            'wait_for_video': (1, 30),
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg': ffmpeg_args
            }
        }
        
        try:
            print("Starting ultra-compressed recording (HEVC)...")
            self.recording = True
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.recording = False

# Usage examples
if __name__ == "__main__":
    recorder = RealtimeCompressRecorder(output_dir="recording")
    
    stream_url = "https://www.youtube.com/watch?v=0_DjDdfqtUE"
    # stream_url = "https://www.youtube.com/watch?v=51iONeETSng"
    '''
    # Record with target file size
    print("Recording with 500MB target size...")
    recorder.record_with_realtime_compression(
        stream_url, 
        duration_minutes=1, 
        target_filesize_mb=500
    )
    '''
    # Record with adaptive compression
    print("\nRecording with adaptive compression...")
    recorder.record_adaptive_compression(stream_url, duration_minutes=180)
    '''
    # Record with ultra compression
    print("\nRecording with ultra compression...")
    recorder.record_ultra_compressed(stream_url, duration_minutes=1)
    '''
