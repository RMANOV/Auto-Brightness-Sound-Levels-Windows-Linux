#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import platform
import sys
from threading import Thread, Event, Lock
from queue import Queue, Empty
from typing import Optional, Tuple, cast
import re

# Gracefully handle optional dependencies
try:
    from numba import njit  # type: ignore
except ImportError:
    print("Warning: numba not found. Using fallback implementation.")
    # Define a fallback decorator if numba is not available
    def njit(func):
        return func

# Different audio detection methods for Linux
AUDIO_AVAILABLE = False
AUDIO_METHOD = ""

# Try using sounddevice if available
try:
    import sounddevice as sd  # type: ignore
    AUDIO_AVAILABLE = True
    AUDIO_METHOD = "sounddevice"
except (ImportError, OSError) as e:
    print(f"Warning: sounddevice not available - {e}")

# If sounddevice failed, try using ALSA directly as a fallback
if not AUDIO_AVAILABLE:
    has_arecord = os.system("which arecord > /dev/null 2>&1") == 0
    has_sox = os.system("which sox > /dev/null 2>&1") == 0
    
    if has_arecord:
        AUDIO_AVAILABLE = True
        AUDIO_METHOD = "arecord"
        print("Using ALSA arecord for audio capturing")
    elif has_sox:
        AUDIO_AVAILABLE = True
        AUDIO_METHOD = "sox"
        print("Using SoX for audio capturing")
        
if not AUDIO_AVAILABLE:
    print("Warning: Audio features disabled - no working capture method found")

# Try different screen capture methods
SCREEN_CAPTURE_METHOD = ""
SCREEN_CAPTURE_AVAILABLE = False

# On Fedora, try Pillow first since it works better
try:
    from PIL import ImageGrab  # type: ignore
    # Test if ImageGrab actually works (it might be present but not functional on some systems)
    try:
        test_grab = ImageGrab.grab(bbox=(0, 0, 10, 10))  # Small area to test
        test_grab.size  # Access a property to verify it works
        SCREEN_CAPTURE_METHOD = "pillow"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found PIL.ImageGrab for screen content analysis")
    except Exception as e:
        print(f"Warning: PIL.ImageGrab is installed but not functional: {e}")
except ImportError:
    print("Warning: PIL.ImageGrab not found. Trying alternative methods.")

# Try MSS as backup option
if not SCREEN_CAPTURE_AVAILABLE:
    try:
        import mss  # type: ignore
        SCREEN_CAPTURE_METHOD = "mss"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found MSS for screen content analysis")
    except ImportError:
        print("Warning: MSS not found.")

# Try xrandr/import method which works on most Linux systems
if not SCREEN_CAPTURE_AVAILABLE:
    import subprocess
    import tempfile
    import os
    
    # Check if we have the necessary tools
    has_xrandr = os.system("which xrandr > /dev/null 2>&1") == 0
    has_import = os.system("which import > /dev/null 2>&1") == 0
    
    if has_xrandr and has_import:
        SCREEN_CAPTURE_METHOD = "xrandr-import"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found xrandr/import tools for screen content analysis")
        
        # Create a temp directory for screenshots if needed
        os.makedirs(os.path.expanduser("~/.cache/adaptive-controller"), exist_ok=True)

# Try GTK screenshot if others fail
if not SCREEN_CAPTURE_AVAILABLE:
    try:
        import gi  # type: ignore
        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk  # type: ignore
        SCREEN_CAPTURE_METHOD = "gtk"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found GTK for screen content analysis")
    except (ImportError, ValueError):
        print("Warning: GTK screenshot method not available.")

if not SCREEN_CAPTURE_AVAILABLE:
    print("Screen content analysis disabled - no working method found.")


class AdaptiveBrightnessVolumeController:
    def __init__(self, camera_index: int = 0,
                 brightness_range: Tuple[int, int] = (5, 45),
                 volume_range: Tuple[int, int] = (3, 35)):
        self.system = platform.system().lower()
        if self.system not in ["linux"]:
            print(f"Currently only Linux is supported. Detected: {self.system}")
            print("Support for Windows and other Linux distributions coming soon")
            sys.exit(1)
            
        # Check for brightness control tools
        self.brightness_method = self._detect_brightness_method()
        if not self.brightness_method:
            print("Error: No supported brightness control method found!")
            print("Options:")
            print("  1. Install brightnessctl: sudo dnf install brightnessctl")
            print("  2. Install xbacklight: sudo dnf install xbacklight")
            print("  3. Make sure /sys/class/backlight/ is accessible")
            sys.exit(1)

        # Configuration
        self.camera_index: int = camera_index
        self.min_brightness: int = brightness_range[0]
        self.max_brightness: int = brightness_range[1]
        self.min_volume: int = volume_range[0]
        self.max_volume: int = volume_range[1]

        # Initialize state
        self.cap: Optional[cv2.VideoCapture] = None
        self.setup_camera()
        self.setup_state()

        # Threading and synchronization
        self.stop_event: Event = Event()
        self.lock: Lock = Lock()

        # Activity tracking
        self.last_activity_time: float = time.time()
        self.is_active: bool = True
        self.inactivity_threshold: int = 300  # seconds
        self.inactivity_check_interval: float = 1.0  # seconds

        # Smoothing parameters
        self.brightness_smoothing_factor: float = 0.3
        self.volume_smoothing_factor: float = 0.2
        self.brightness_change_threshold: float = 5.0

        # Screen content analysis
        self.screen_check_interval: float = 2.0  # seconds (increased to reduce errors)
        self.last_screen_check_time: float = 0.0
        self.screen_brightness_factor: float = 1.0
        self.screen_capture_error_count: int = 0
        self.max_screen_errors: int = 5  # Show only first few errors
        
        # Initialize screen capture based on available method
        self.sct = None
        self.pil_available = SCREEN_CAPTURE_METHOD == "pillow"
        self.gtk_available = SCREEN_CAPTURE_METHOD == "gtk"
        self.xrandr_available = SCREEN_CAPTURE_METHOD == "xrandr-import"
        self.screenshot_path = os.path.expanduser("~/.cache/adaptive-controller/screenshot.png")
        
        if SCREEN_CAPTURE_METHOD == "mss":
            try:
                self.sct = mss.mss()
                print("Using MSS for screen content analysis")
            except Exception as e:
                print(f"Failed to initialize MSS: {e}")
                SCREEN_CAPTURE_AVAILABLE = False

        # Audio settings
        self.audio_duration: float = 0.1  # seconds
        self.audio_samplerate: int = 44100  # Hz
        
        # Noise level thresholds - adjusted for more comfortable volume range
        # Typical ambient room noise is around 1e-4 to 5e-4
        # Conversation/music might be around 1e-3 to 5e-3
        # Loud environments can be above 1e-2
        self.min_noise_level: float = 5e-6  # Very quiet environment
        self.max_noise_level: float = 8e-3  # Fairly loud environment

        # State variables
        self.current_brightness: float = 30.0
        self.smoothed_brightness: float = 30.0
        self.prev_camera_brightness: Optional[float] = None
        self.current_volume: float = 40.0
        self.smoothed_volume: float = 40.0
        
        # Warmup parameters to avoid initial spike
        self.warmup_frames: int = 20  # For brightness
        self.current_warmup_frame: int = 0
        self.warmup_cooldown: float = 0.05  # For brightness transitions (was 0.2)
        self.is_in_warmup: bool = True
        self.initial_brightness: Optional[float] = None
        self.initial_volume: Optional[float] = None
        
        # Separate audio warmup parameters (more extreme to prevent audio shock)
        self.audio_warmup_frames: int = 40  # Double the frames for audio warmup
        self.audio_warmup_cooldown: float = 0.005  # 10x slower transitions than brightness
        self.is_audio_in_warmup: bool = True  # Separate flag for audio warmup
        self.audio_warmup_threshold: int = 10  # Additional grace period after main warmup
        
        # Real brightness calibration - used to track actual screen vs reported values
        self.brightness_calibration_factor: float = 0.4  # Scale factor to match real values
        self.last_significant_change_time: float = 0.0
        self.sensitivity_to_changes: float = 1.5  # Increase sensitivity to light changes

    def setup_camera(self):
        """Initialize camera with fallback to other available cameras"""
        # Try the specified camera index first
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # If the specified camera doesn't work, try other indices
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_index}, trying alternatives...")
            
            # Try camera indices 0-9
            for idx in range(10):
                if idx == self.camera_index:
                    continue  # Skip the one we already tried
                    
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    print(f"Found working camera at index {idx}")
                    self.cap = test_cap
                    self.camera_index = idx
                    break
                else:
                    test_cap.release()
        
        # If we have a working camera, configure it
        if self.cap is not None and self.cap.isOpened():
            # Reduce camera resolution for performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            print(f"Camera initialized at index {self.camera_index}")
        else:
            print("No working camera found")
            msg = "Using fallback brightness control without ambient sensing"
            print(msg)
            self.cap = cast(Optional[cv2.VideoCapture], None)

    def load_saved_state(self) -> Tuple[Optional[float], Optional[float]]:
        """Load previously saved brightness and volume settings"""
        try:
            config_file = os.path.expanduser("~/.config/adaptive-controller/last_state.txt")
            if os.path.exists(config_file):
                # Read saved settings
                brightness = None
                volume = None
                timestamp = None
                
                with open(config_file, "r") as f:
                    for line in f:
                        if line.startswith("brightness="):
                            brightness = float(line.strip().split("=")[1])
                        elif line.startswith("volume="):
                            volume = float(line.strip().split("=")[1])
                        elif line.startswith("timestamp="):
                            timestamp = int(line.strip().split("=")[1])
                
                # Check if settings are not too old (max 24 hours)
                if timestamp and time.time() - timestamp < 24 * 60 * 60:
                    return brightness, volume
        except Exception as e:
            print(f"Warning: Could not load saved settings: {e}")
            
        return None, None
        
    def setup_state(self) -> None:
        # Try to load saved settings
        saved_brightness, saved_volume = self.load_saved_state()
        
        # Setup brightness
        try:
            self.current_brightness = self.get_brightness()
        except Exception:
            if saved_brightness is not None:
                print(f"Using saved brightness: {saved_brightness}%")
                self.current_brightness = saved_brightness
            else:
                self.current_brightness = 30.0
        
        self.smoothed_brightness = self.current_brightness
        self.prev_camera_brightness = None

        # Setup volume
        try:
            self.current_volume = float(self.get_volume())
        except Exception:
            if saved_volume is not None:
                print(f"Using saved volume: {saved_volume}%")
                self.current_volume = saved_volume
            else:
                self.current_volume = self.min_volume + (self.max_volume - self.min_volume) / 2
        
        self.smoothed_volume = self.current_volume

    def on_activity(self) -> None:
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
            self.inactivity_check_interval = 1.0
            self.stop_event.clear()

    def on_inactivity(self) -> None:
        self.is_active = False
        if self.cap:
            self.cap.release()
            self.cap = cast(Optional[cv2.VideoCapture], None)
        cv2.destroyAllWindows()

    @staticmethod
    @njit
    def calculate_brightness(frame: np.ndarray) -> float:
        return np.mean(frame) / 255 * 100

    def analyze_image(self, frame: Optional[np.ndarray]) -> float:
        if frame is None:
            return 50.0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.calculate_brightness(gray_frame)

    def analyze_screen_content(self) -> float:
        global SCREEN_CAPTURE_AVAILABLE
        if not SCREEN_CAPTURE_AVAILABLE:
            return 1.0  # Neutral adjustment if screen capture not available

        try:
            # Capture screen based on available method
            if self.pil_available:
                try:
                    # PIL is more reliable on most Linux systems
                    screenshot = ImageGrab.grab()
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"PIL capture failed: {e}")
            elif self.xrandr_available:
                try:
                    # Use xrandr+import for reliable screen capture on X11 systems
                    # First capture a small part of the screen to save resources
                    # Get screen size first
                    process = subprocess.run(
                        ["xrandr", "--current"], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    output = process.stdout
                    
                    # Parse the primary display resolution
                    lines = output.strip().split('\n')
                    resolution = None
                    for line in lines:
                        if "*" in line and "+" in line:  # active mode with position
                            parts = line.split()
                            for part in parts:
                                if "x" in part and part[0].isdigit():
                                    resolution = part
                                    break
                            if resolution:
                                break
                                
                    if not resolution:
                        # Fallback to a reasonable resolution
                        resolution = "1920x1080"
                    
                    width, height = map(int, resolution.split("x"))
                    
                    # Take a screenshot of center region (1/4 of screen)
                    center_x = width // 4
                    center_y = height // 4
                    center_width = width // 2
                    center_height = height // 2
                    
                    # Use imagemagick's import to capture screen
                    cmd = f"import -window root -crop {center_width}x{center_height}+{center_x}+{center_y} {self.screenshot_path}"
                    result = os.system(cmd)
                    
                    if result != 0:
                        raise Exception("Failed to capture screenshot with import")
                        
                    # Read the screenshot
                    img = cv2.imread(self.screenshot_path)
                    if img is None:
                        raise Exception("Failed to read captured screenshot")
                        
                except Exception as e:
                    raise Exception(f"xrandr-import capture failed: {e}")
            elif self.gtk_available:
                try:
                    window = Gdk.get_default_root_window()
                    x, y, width, height = window.get_geometry()
                    pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
                    img = np.array(pb.get_pixels_array())
                except Exception as e:
                    raise Exception(f"GTK capture failed: {e}")
            elif SCREEN_CAPTURE_METHOD == "mss" and self.sct is not None:
                try:
                    monitor = self.sct.monitors[1]  # Primary monitor
                    screenshot = self.sct.grab(monitor)
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"MSS capture failed: {e}")
            else:
                return 1.0  # No working method

            # Calculate brightness
            brightness = np.mean(img) / 255
            self.screen_capture_error_count = 0  # Reset error count on success

            # Adjust brightness factor based on screen content - INVERTED LOGIC
            # For bright content, we want to INCREASE brightness to improve visibility
            # For dark content, we want to DECREASE brightness to reduce eye strain
            if brightness > 0.7:  # Very bright content (white documents, etc)
                return 1.2  # Increase screen brightness for better visibility
            elif brightness < 0.3:  # Dark content (dark mode apps, etc)
                return 0.8  # Reduce screen brightness for comfort with dark content
            else:
                return 1.0  # Neutral adjustment
        except Exception as e:
            # Limit error messages to avoid spam
            if self.screen_capture_error_count < self.max_screen_errors:
                print(f"Screen analysis error: {e}")
                self.screen_capture_error_count += 1
            elif self.screen_capture_error_count == self.max_screen_errors:
                print("Too many screen capture errors, suppressing further messages")
                self.screen_capture_error_count += 1
            
            # After too many errors, try to disable the problematic method
            if self.screen_capture_error_count > self.max_screen_errors + 10:
                print(f"Warning: Disabling problematic screen capture method: {SCREEN_CAPTURE_METHOD}")
                SCREEN_CAPTURE_AVAILABLE = False
                
            return 1.0

    def _detect_brightness_method(self) -> str:
        """Detect available brightness control method"""
        # Check for brightnessctl
        if os.system("which brightnessctl > /dev/null 2>&1") == 0:
            return "brightnessctl"
            
        # Check for xbacklight
        if os.system("which xbacklight > /dev/null 2>&1") == 0:
            return "xbacklight"
            
        # Check for direct sys file access
        backlight_dirs = os.listdir("/sys/class/backlight") if os.path.exists("/sys/class/backlight") else []
        if backlight_dirs:
            self.backlight_dir = f"/sys/class/backlight/{backlight_dirs[0]}"
            try:
                with open(f"{self.backlight_dir}/brightness", "r") as f:
                    f.read()
                with open(f"{self.backlight_dir}/max_brightness", "r") as f:
                    f.read()
                return "sysfs"
            except (IOError, PermissionError):
                pass
                
        return ""
        
    def get_brightness(self) -> float:
        """Get current screen brightness as percentage"""
        if self.brightness_method == "brightnessctl":
            brightness = os.popen("brightnessctl get").read().strip()
            max_brightness = os.popen("brightnessctl max").read().strip()
            return float(brightness) / float(max_brightness) * 100
            
        elif self.brightness_method == "xbacklight":
            brightness = os.popen("xbacklight -get").read().strip()
            return float(brightness)
            
        elif self.brightness_method == "sysfs":
            try:
                with open(f"{self.backlight_dir}/brightness", "r") as f:
                    brightness = int(f.read().strip())
                with open(f"{self.backlight_dir}/max_brightness", "r") as f:
                    max_brightness = int(f.read().strip())
                return brightness / max_brightness * 100
            except (IOError, ValueError) as e:
                print(f"Error reading brightness: {e}")
                return 50.0
                
        return 50.0  # Fallback

    def set_brightness(self, brightness: float) -> None:
        """Set screen brightness as percentage"""
        # Apply limits to input value
        brightness = max(self.min_brightness, min(self.max_brightness, brightness))
        
        # NO LONGER applying calibration factor automatically
        # Instead, we'll apply calibration ONLY if we detect the value is too low
        
        # First try to use the requested value directly
        calibrated_brightness = round(brightness)
        
        # If we suspect we're in an environment where reported != actual:
        # Comment this out to disable calibration adjustment completely
        # calibrated_brightness = max(5, round(brightness / self.brightness_calibration_factor))
        
        # ALWAYS enforce the range limits regardless of calibration
        calibrated_brightness = max(self.min_brightness, 
                               min(self.max_brightness, calibrated_brightness))
        
        # More detailed debug output
        print(f"Brightness: Target: {brightness:.1f}% → Setting: {calibrated_brightness}%")
        
        if self.brightness_method == "brightnessctl":
            os.system(f"brightnessctl set {calibrated_brightness}%")
            
        elif self.brightness_method == "xbacklight":
            os.system(f"xbacklight -set {calibrated_brightness}")
            
        elif self.brightness_method == "sysfs":
            try:
                with open(f"{self.backlight_dir}/max_brightness", "r") as f:
                    max_brightness = int(f.read().strip())
                    
                # Convert percentage to absolute value
                value = int((brightness / 100) * max_brightness)
                
                # Write the new brightness value
                try:
                    with open(f"{self.backlight_dir}/brightness", "w") as f:
                        f.write(str(value))
                except PermissionError:
                    # Try with sudo if direct write fails
                    os.system(f"echo {value} | sudo tee {self.backlight_dir}/brightness > /dev/null")
            except Exception as e:
                print(f"Error setting brightness: {e}")

    def get_volume(self) -> int:
        """Get current volume level as percentage"""
        try:
            # Try amixer first
            if os.system("which amixer > /dev/null 2>&1") == 0:
                output = os.popen("amixer get Master").read()
                matches = re.search(r'\[([0-9]+)%\]', output)
                if matches:
                    return int(matches.group(1))
            
            # Try pactl (PulseAudio)
            if os.system("which pactl > /dev/null 2>&1") == 0:
                output = os.popen("pactl list sinks | grep Volume").read()
                matches = re.search(r'(\d+)%', output)
                if matches:
                    return int(matches.group(1))
                    
            # Try wpctl (Pipewire)
            if os.system("which wpctl > /dev/null 2>&1") == 0:
                output = os.popen("wpctl get-volume @DEFAULT_AUDIO_SINK@").read()
                matches = re.search(r'Volume: ([0-9.]+)', output)
                if matches:
                    volume_float = float(matches.group(1))
                    return int(volume_float * 100)
                    
            return 40  # Default if no method worked
        except Exception as e:
            print(f"Error getting volume: {e}")
            return 40

    def set_volume(self, volume: float) -> None:
        """Set volume level as percentage"""
        volume = max(self.min_volume, min(self.max_volume, int(volume)))
        
        # Try multiple methods in sequence until one works
        success = False
        
        # Method 1: amixer
        if not success and os.system("which amixer > /dev/null 2>&1") == 0:
            exit_code = os.system(f"amixer set Master {volume}% > /dev/null 2>&1")
            success = (exit_code == 0)
            
        # Method 2: pactl (PulseAudio)
        if not success and os.system("which pactl > /dev/null 2>&1") == 0:
            exit_code = os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {volume}% > /dev/null 2>&1")
            success = (exit_code == 0)
            
        # Method 3: wpctl (PipeWire)
        if not success and os.system("which wpctl > /dev/null 2>&1") == 0:
            volume_float = volume / 100.0
            exit_code = os.system(f"wpctl set-volume @DEFAULT_AUDIO_SINK@ {volume_float} > /dev/null 2>&1")
            success = (exit_code == 0)
            
        if not success:
            print(f"Warning: Failed to set volume to {volume}%")

    def capture_audio(self) -> np.ndarray:
        if not AUDIO_AVAILABLE:
            # Return empty array if audio capture is not available
            return np.zeros(int(self.audio_duration * self.audio_samplerate))
            
        try:
            sample_count = int(self.audio_duration * self.audio_samplerate)
            
            if AUDIO_METHOD == "sounddevice":
                # Use sounddevice
                audio = sd.rec(sample_count,
                               samplerate=self.audio_samplerate,
                               channels=1,
                               blocking=True)
                return audio.flatten()
                
            elif AUDIO_METHOD == "arecord":
                # Use ALSA arecord
                audio_file = os.path.expanduser("~/.cache/adaptive-controller/audio.wav")
                duration_ms = int(self.audio_duration * 1000)
                cmd = f"arecord -q -d {self.audio_duration} -f S16_LE -r {self.audio_samplerate} -c1 {audio_file}"
                result = os.system(cmd)
                
                if result != 0:
                    raise Exception("Failed to record audio with arecord")
                    
                # Use OpenCV to read audio file
                import wave
                with wave.open(audio_file, 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
                    
            elif AUDIO_METHOD == "sox":
                # Use SoX for recording
                audio_file = os.path.expanduser("~/.cache/adaptive-controller/audio.wav")
                cmd = f"sox -n -r {self.audio_samplerate} -c 1 {audio_file} trim 0 {self.audio_duration}"
                result = os.system(cmd)
                
                if result != 0:
                    raise Exception("Failed to record audio with sox")
                    
                # Read audio file
                import wave
                with wave.open(audio_file, 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
            
            # Fallback to empty array
            return np.zeros(int(self.audio_duration * self.audio_samplerate))
            
        except Exception as e:
            print(f"Audio capture error: {e}")
            return np.zeros(int(self.audio_duration * self.audio_samplerate))

    def compute_noise_level(self, audio: np.ndarray) -> float:
        return np.sqrt(np.mean(np.square(audio)))

    def process_frames(self, frame_queue: Queue, brightness_queue: Queue) -> None:
        while not self.stop_event.is_set():
            if self.is_active and self.cap and self.cap.isOpened():
                try:
                    frame = frame_queue.get(timeout=1)
                    brightness = self.analyze_image(frame)
                    brightness_queue.put(brightness)
                except Empty:
                    continue
            else:
                time.sleep(self.inactivity_check_interval)

    def run(self) -> None:
        frame_queue: Queue[np.ndarray] = Queue(maxsize=10)
        brightness_queue: Queue[float] = Queue(maxsize=10)

        # Start frame processing thread if camera is available
        if self.cap and self.cap.isOpened():
            process_thread = Thread(target=self.process_frames, args=(frame_queue, brightness_queue))
            process_thread.daemon = True
            process_thread.start()

        update_interval = 0.5
        last_brightness_change_time = time.time()

        try:
            while not self.stop_event.is_set():
                current_time = time.time()

                # Check for inactivity
                if current_time - self.last_activity_time > self.inactivity_threshold:
                    self.on_inactivity()

                if self.is_active:
                    # Screen content analysis (less frequent)
                    if SCREEN_CAPTURE_AVAILABLE and current_time - self.last_screen_check_time > self.screen_check_interval:
                        self.screen_brightness_factor = self.analyze_screen_content()
                        self.last_screen_check_time = current_time

                    # Ambient light detection via camera
                    camera_brightness = None
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            frame_queue.put(frame)
                            try:
                                camera_brightness = brightness_queue.get(block=False)
                                self.prev_camera_brightness = camera_brightness
                            except Empty:
                                camera_brightness = self.prev_camera_brightness
                    else:
                        camera_brightness = self.prev_camera_brightness

                    # Determine target brightness based on ambient light and screen content
                    if camera_brightness is not None:
                        # INVERTED LOGIC: In dark room, we want lower brightness, in bright room - higher
                        # We want to respond more dramatically to environment changes
                        
                        # Improved tracking of light changes with better detection
                        last_camera_brightness = self.prev_camera_brightness
                        if last_camera_brightness is not None:
                            # Much more sensitive detection of changes (especially dimming)
                            brightness_change = camera_brightness - last_camera_brightness
                            abs_change = abs(brightness_change)
                            
                            # Lower threshold for dimming (going from bright to dark)
                            dimming_threshold = 8  # Detect dimming changes faster
                            brightening_threshold = 12  # For going from dark to bright
                            
                            # Use appropriate threshold based on direction of change
                            if (brightness_change < 0 and abs_change > dimming_threshold) or \
                               (brightness_change > 0 and abs_change > brightening_threshold):
                                self.last_significant_change_time = current_time
                                
                                # Print detailed information about the light change
                                direction = "DIMMING ⬇️" if brightness_change < 0 else "BRIGHTENING ⬆️"
                                print(f"Light change detected - {direction}: {last_camera_brightness:.1f} → {camera_brightness:.1f} (Δ{brightness_change:.1f})")
                                
                                # For dimming, apply a boost to make it respond faster
                                if brightness_change < 0:
                                    self.sensitivity_to_changes = 2.5  # Higher boost for dimming
                                else:
                                    self.sensitivity_to_changes = 1.5  # Normal boost for brightening
                        
                        # Enhanced formula that responds better to environment changes
                        # Print current camera brightness for debugging
                        if self.current_warmup_frame % 10 == 0 or current_time - self.last_significant_change_time < 5:
                            print(f"Camera brightness: {camera_brightness:.1f}")
                            
                        # DEFINITIVE RULE: 
                        # ✓ Dark room + dark content = lowest brightness (5%)
                        # ✓ Bright room + bright content = highest brightness (45%)
                        #
                        # Direct mapping (NOT inverted anymore):
                        # Camera brightness 0-30: very dark room = low screen brightness (5-15%)
                        # Camera brightness 30-70: medium room = medium brightness (15-35%)
                        # Camera brightness 70-100: bright room = high brightness (35-45%)
                        
                        # We apply DIRECT mapping for more intuitive behavior
                        # Higher camera brightness = higher screen brightness  
                        
                        # INCREASED reaction speed by 50% as requested
                        # 1.5 = original rate + 50% increase
                        base_adjustment_speed = 1.5
                        
                        # Check for significant light changes to react even faster
                        time_since_change = current_time - self.last_significant_change_time
                        if time_since_change < 5:  # Recent significant change (last 5 seconds)
                            # Boost speed by 2x during transitions (3x normal speed)
                            adjustment_boost = 2.0 * base_adjustment_speed
                            print(f"⚡ Fast adjustment mode - boosted speed: {adjustment_boost:.1f}x normal")
                        else:
                            # Normal 50% faster speed
                            adjustment_boost = base_adjustment_speed
                        
                        # DIRECT CALCULATION BASED ON CAMERA BRIGHTNESS
                        # Using simple linear mapping: 
                        # 0 camera brightness → 5% screen brightness
                        # 100 camera brightness → 45% screen brightness
                        
                        # Slope calculation for y = mx + b line:
                        # m = (y2-y1)/(x2-x1) = (45-5)/(100-0) = 0.4
                        # Direct linear formula: brightness = 5 + (camera_brightness * 0.4)
                        
                        # Calculate screen brightness directly from camera brightness
                        # DIRECT MAPPING FOR CAMERA TO BRIGHTNESS
                        # For camera 45% to yield screen brightness 30%:
                        
                        # Let's use a different approach - apply a direct mapping formula 
                        # instead of enhancing the value and then applying range formulas
                        
                        # This formula directly maps camera brightness to screen brightness:
                        # Linear scaling: camera 0% → screen 5%, camera 100% → screen 45%
                        # But with a boost in the middle range to ensure 45% camera → 30% screen
                        
                        # Base linear scaling formula
                        base_linear = 5 + (camera_brightness * 0.4)  # Simple linear: 0→5%, 100→45%
                        
                        # Apply a curve to boost middle values
                        # This applies extra boost to values around 45% camera brightness
                        boost_factor = 1.0
                        if 35 <= camera_brightness <= 55:
                            # Maximum boost at 45%, tapering off at 35% and 55%
                            distance_from_45 = abs(camera_brightness - 45)
                            # Boost up to 35% (multiplicative factor from 1.0 to 1.35)
                            max_boost = 1.35  # 35% boost at center point (45% camera)
                            boost_factor = max_boost - (distance_from_45 / 10 * (max_boost - 1.0))
                        
                        # Apply the boost to get our target brightness
                        target_brightness = base_linear * boost_factor
                        
                        # Ensure it stays within our min/max range
                        target_brightness = max(self.min_brightness, min(self.max_brightness, target_brightness))
                        
                        # Print detailed information about the calculation
                        if camera_brightness < 30:
                            print(f"🌙 Dark room - target: {target_brightness:.1f}% (camera: {camera_brightness:.1f}, boost: {boost_factor:.2f}x)")
                        elif camera_brightness > 70:
                            print(f"☀️ Bright room - target: {target_brightness:.1f}% (camera: {camera_brightness:.1f}, boost: {boost_factor:.2f}x)")
                        else:
                            print(f"🌤️ Medium light - target: {target_brightness:.1f}% (camera: {camera_brightness:.1f}, boost: {boost_factor:.2f}x)")
                        
                        # Apply additional screen content analysis if available
                        if SCREEN_CAPTURE_AVAILABLE:
                            target_brightness = target_brightness * self.screen_brightness_factor

                        # Handle warmup period to avoid initial spikes
                        if self.is_in_warmup:
                            # During the first several frames, start from current brightness and move very gradually
                            self.current_warmup_frame += 1
                            
                            # Store the actual current brightness as starting point (not a default value)
                            if self.initial_brightness is None:
                                try:
                                    # Try to get the real current brightness from the system
                                    self.initial_brightness = self.get_brightness()
                                    # Start exactly where we are now
                                    self.smoothed_brightness = self.initial_brightness
                                    print(f"Starting from current brightness: {self.initial_brightness}%")
                                except Exception:
                                    self.initial_brightness = self.smoothed_brightness
                                    print(f"Using default initial brightness: {self.initial_brightness}%")
                            
                            # Show progress during extended warmup
                            if self.current_warmup_frame <= self.warmup_frames:
                                if self.current_warmup_frame % 4 == 0:  # Less frequent progress updates
                                    print(f"Calibrating... {(self.current_warmup_frame * 100) // self.warmup_frames}%")
                                
                                # Extra gentle transitions during warmup (very small changes per frame)
                                # This avoids the initial jump by making incredibly slow adjustments
                                error = target_brightness - self.smoothed_brightness
                                self.smoothed_brightness += error * (self.brightness_smoothing_factor * self.warmup_cooldown)
                            else:
                                # End of warmup period
                                self.is_in_warmup = False
                                print("Calibration complete, applying normal brightness control")
                                # Remember this as our baseline brightness after warmup
                                self.prev_camera_brightness = camera_brightness
                        else:
                            # Normal operation - ALWAYS apply changes, but smooth the transition
                            error = target_brightness - self.smoothed_brightness
                            
                            # Apply the increased adjustment speed (50% faster + boost during changes)
                            # Either base_adjustment_speed (1.5) or with additional boost during transitions
                            smooth_factor = self.brightness_smoothing_factor * adjustment_boost
                            self.smoothed_brightness += error * smooth_factor
                            
                            # Debug output if significant changes are happening
                            if abs(error) > 2.0:
                                print(f"Adjusting brightness: {self.smoothed_brightness:.1f}% → {target_brightness:.1f}% " +
                                      f"(change rate: {smooth_factor:.2f}, step: {error * smooth_factor:.2f})")
                                
                        # Apply limits to brightness
                        self.smoothed_brightness = max(
                            float(self.min_brightness),
                            min(float(self.max_brightness),
                                self.smoothed_brightness))

                        # Apply the new brightness (always, but with different smoothing rates)
                        try:
                            self.set_brightness(round(self.smoothed_brightness))
                            self.current_brightness = self.smoothed_brightness
                            last_brightness_change_time = current_time
                        except Exception as e:
                            print(f"Brightness setting error: {e}")

                    # Audio processing and volume adjustment
                    if AUDIO_AVAILABLE:
                        audio = self.capture_audio()
                        noise_level = self.compute_noise_level(audio)

                        # Map noise level to volume percentage with adaptive curve
                        noise_range = self.max_noise_level - self.min_noise_level
                        normalized_noise = (noise_level - self.min_noise_level) / noise_range
                        normalized_noise = max(0.0, min(1.0, normalized_noise))
                        
                        # Apply a logarithmic curve to make the volume response more natural
                        # Human hearing perception is roughly logarithmic
                        if normalized_noise > 0:
                            # Mapping to the narrower 3-35% range with logarithmic adjustment
                            # This makes quieter sounds result in lower volumes and 
                            # prevents loud sounds from being too loud
                            
                            # Enhanced volume calculation to be higher than direct correlation
                            # Increase curve factor for higher volume response
                            curve_factor = 0.55  # Increased from 0.45 for higher overall volume
                            
                            # Adjusted multiplier for better response at low noise levels
                            multiplier = 12  # Increased from 10 for stronger response
                            
                            # Increased bias factor for a higher base volume
                            # The goal is to get around 25-30% volume in quiet rooms instead of 20%
                            bias = 0.22  # Increased from 0.15 for higher overall volume
                            
                            # Calculate adjusted noise level with enhanced curve for higher values
                            # Power function to boost lower values more (similar to brightness enhancement)
                            normalized_noise_enhanced = normalized_noise**0.8 * 1.2
                            normalized_noise_enhanced = min(normalized_noise_enhanced, 1.0)
                            
                            # Apply enhanced formula with logarithmic curve
                            adjusted_noise = curve_factor * np.log10(1 + multiplier * normalized_noise_enhanced) + bias
                            
                            # Ensure the adjusted value stays between 0-1
                            adjusted_noise = max(0.0, min(1.0, adjusted_noise))
                        else:
                            # Base level adjustment for complete silence
                            adjusted_noise = 0.22  # Increased from 0.15 to give approximately 25-30% volume
                            
                        volume_range = self.max_volume - self.min_volume
                        target_volume = adjusted_noise * volume_range + self.min_volume

                        # Handle warmup period for volume to avoid initial spikes
                        # Audio has its own separate (and longer) warmup period
                        if self.is_audio_in_warmup:
                            # Store initial values - we want to EXACTLY maintain
                            # the initial volume for several seconds
                            if self.initial_volume is None:
                                try:
                                    # Get the actual current system volume
                                    current_system_volume = float(self.get_volume())
                                    self.initial_volume = current_system_volume
                                    self.smoothed_volume = current_system_volume
                                    print(f"Initial volume locked at: {self.initial_volume}%")
                                except:
                                    self.initial_volume = self.smoothed_volume
                                    print(f"Using default initial volume: {self.initial_volume}%")
                            
                            # First phase: Stay at EXACTLY current volume for 'audio_warmup_threshold' frames
                            if self.current_warmup_frame <= self.audio_warmup_threshold:
                                # Ignore all target volumes and keep exactly where we are
                                self.smoothed_volume = self.initial_volume
                                volume_change = 0  # No change at all
                            # Second phase: Very gradual transition over extended period
                            elif self.current_warmup_frame <= self.audio_warmup_frames:
                                if self.current_warmup_frame % 8 == 0:  # Less frequent updates
                                    audio_progress = ((self.current_warmup_frame - self.audio_warmup_threshold) * 100) // (self.audio_warmup_frames - self.audio_warmup_threshold)
                                    print(f"Audio calibrating... {audio_progress}%")
                                
                                # Ultra-smooth transition - 100x slower than normal
                                volume_error = target_volume - self.smoothed_volume
                                volume_change = volume_error * (self.volume_smoothing_factor * self.audio_warmup_cooldown)
                                self.smoothed_volume += volume_change
                            else:
                                # End of audio warmup
                                self.is_audio_in_warmup = False
                                print("Audio calibration complete")
                                # Apply normal adjustment, but still gentler than standard
                                volume_error = target_volume - self.smoothed_volume
                                volume_change = volume_error * (self.volume_smoothing_factor * 0.5)  # 50% normal speed
                                self.smoothed_volume += volume_change
                        else:
                            # Normal volume adjustment after warmup
                            volume_error = target_volume - self.smoothed_volume
                            volume_change = volume_error * self.volume_smoothing_factor
                            self.smoothed_volume += volume_change
                            
                        # Debug info about volume changes
                        if abs(volume_change) > 0.1:
                            print(f"Volume: {self.smoothed_volume:.1f}% (change: {volume_change:.2f})")

                        # Apply volume limits
                        min_vol = float(self.min_volume)
                        max_vol = float(self.max_volume)
                        self.smoothed_volume = max(min_vol, min(max_vol, self.smoothed_volume))

                        # Apply the new volume
                        try:
                            self.set_volume(round(self.smoothed_volume))
                            self.current_volume = self.smoothed_volume
                        except Exception as e:
                            print(f"Volume setting error: {e}")

                    # Adaptive polling interval
                    if current_time - last_brightness_change_time > 10:
                        # No significant changes recently, can slow down polling
                        update_interval = min(update_interval * 1.2, 2.0)
                    else:
                        # Recent changes, need more responsive polling
                        update_interval = max(update_interval / 1.2, 0.1)

                    time.sleep(update_interval)
                else:
                    time.sleep(self.inactivity_check_interval)
                    # Check if we should become active again
                    self.on_activity()

                    # Recreate camera if we're active again
                    if self.is_active and self.cap is None:
                        self.setup_camera()

        except KeyboardInterrupt:
            print("Stopping controller...")
        finally:
            # Save current brightness and volume settings before exiting
            try:
                # Create configuration directory if it doesn't exist
                config_dir = os.path.expanduser("~/.config/adaptive-controller")
                os.makedirs(config_dir, exist_ok=True)
                
                # Save settings
                with open(f"{config_dir}/last_state.txt", "w") as f:
                    f.write(f"brightness={round(self.smoothed_brightness)}\n")
                    f.write(f"volume={round(self.smoothed_volume)}\n")
                    f.write(f"timestamp={int(time.time())}\n")
                    
                print(f"Settings saved to {config_dir}/last_state.txt")
            except Exception as e:
                print(f"Warning: Could not save settings: {e}")
                
            self.stop_event.set()
            if self.cap:
                self.cap.release()
                
            # Safely destroy OpenCV windows if possible
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # OpenCV may not be compiled with GTK support
                pass


if __name__ == '__main__':
    # Make sure cache directories exist
    os.makedirs(os.path.expanduser("~/.cache/adaptive-controller"), exist_ok=True)

    # Check for audio dependencies if missing
    if not AUDIO_AVAILABLE:
        print("\nAudio features are disabled. To enable audio support:")
        print("  For sounddevice: sudo dnf install portaudio-devel && pip install sounddevice --user")
        print("  Alternative methods: sudo dnf install alsa-utils sox")
        print("\nContinuing without audio features...\n")
    
    # Check for PIL for screen capture
    if not SCREEN_CAPTURE_AVAILABLE:
        print("\nScreen content analysis is disabled. To enable, install one of:")
        print("  pip install mss --user        # Preferred method")
        print("  pip install pillow --user     # Alternative method")
        print("\nContinuing without screen content analysis...\n")
    
    try:
        controller = AdaptiveBrightnessVolumeController()
        print("\nStarting adaptive brightness and volume controller...")
        print(f"Brightness range: {controller.min_brightness}% - {controller.max_brightness}%")
        print(f"Volume range: {controller.min_volume}% - {controller.max_volume}%")
        print("Detected brightness control method:", controller.brightness_method)
        if SCREEN_CAPTURE_AVAILABLE:
            print("Screen capture method:", SCREEN_CAPTURE_METHOD)
        if AUDIO_AVAILABLE:
            print("Audio control: Enabled (adaptive based on ambient noise)")
        else:
            print("Audio control: Disabled")
        print("\nPress Ctrl+C to stop")
        controller.run()
    except KeyboardInterrupt:
        print("\nStopping controller...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
