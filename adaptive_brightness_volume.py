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

try:
    import sounddevice as sd  # type: ignore
    AUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Audio features disabled - {e}")
    AUDIO_AVAILABLE = False

# Try different screen capture methods
SCREEN_CAPTURE_METHOD = ""
SCREEN_CAPTURE_AVAILABLE = False

# Try MSS first
try:
    import mss  # type: ignore
    SCREEN_CAPTURE_METHOD = "mss"
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    print("Warning: mss not found. Trying alternative screen capture methods.")

# Try Pillow/ImageGrab if MSS fails
if not SCREEN_CAPTURE_AVAILABLE:
    try:
        from PIL import ImageGrab  # type: ignore
        SCREEN_CAPTURE_METHOD = "pillow"
        SCREEN_CAPTURE_AVAILABLE = True
    except ImportError:
        print("Warning: PIL.ImageGrab not found.")

# Try GTK screenshot if others fail
if not SCREEN_CAPTURE_AVAILABLE:
    try:
        import gi  # type: ignore
        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk  # type: ignore
        SCREEN_CAPTURE_METHOD = "gtk"
        SCREEN_CAPTURE_AVAILABLE = True
    except (ImportError, ValueError):
        print("Warning: GTK screenshot method not available.")

if not SCREEN_CAPTURE_AVAILABLE:
    print("Screen content analysis disabled - no working method found.")


class AdaptiveBrightnessVolumeController:
    def __init__(self, camera_index: int = 0,
                 brightness_range: Tuple[int, int] = (5, 45),
                 volume_range: Tuple[int, int] = (2, 60)):
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
        self.min_noise_level: float = 1e-5
        self.max_noise_level: float = 1e-2

        # State variables
        self.current_brightness: float = 30.0
        self.smoothed_brightness: float = 30.0
        self.prev_camera_brightness: Optional[float] = None
        self.current_volume: float = 40.0
        self.smoothed_volume: float = 40.0

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

    def setup_state(self) -> None:
        try:
            self.current_brightness = self.get_brightness()
        except Exception:
            self.current_brightness = 30.0
        self.smoothed_brightness = self.current_brightness
        self.prev_camera_brightness = None

        try:
            self.current_volume = float(self.get_volume())
        except Exception:
            self.current_volume = 40.0
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
        if not SCREEN_CAPTURE_AVAILABLE:
            return 1.0  # Neutral adjustment if screen capture not available

        try:
            # Capture screen based on available method
            if SCREEN_CAPTURE_METHOD == "mss" and self.sct is not None:
                try:
                    monitor = self.sct.monitors[1]  # Primary monitor
                    screenshot = self.sct.grab(monitor)
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"MSS capture failed: {e}")
            elif SCREEN_CAPTURE_METHOD == "pillow":
                try:
                    screenshot = ImageGrab.grab()
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"PIL capture failed: {e}")
            elif SCREEN_CAPTURE_METHOD == "gtk":
                try:
                    window = Gdk.get_default_root_window()
                    x, y, width, height = window.get_geometry()
                    pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
                    img = np.array(pb.get_pixels_array())
                except Exception as e:
                    raise Exception(f"GTK capture failed: {e}")
            else:
                return 1.0  # No working method

            # Calculate brightness
            brightness = np.mean(img) / 255
            self.screen_capture_error_count = 0  # Reset error count on success

            # Adjust brightness factor based on screen content
            if brightness > 0.7:  # Very bright content
                return 0.8  # Reduce screen brightness
            elif brightness < 0.3:  # Dark content
                return 1.2  # Increase screen brightness slightly
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
        brightness = max(self.min_brightness, min(self.max_brightness, brightness))
        
        if self.brightness_method == "brightnessctl":
            os.system(f"brightnessctl set {brightness}%")
            
        elif self.brightness_method == "xbacklight":
            os.system(f"xbacklight -set {brightness}")
            
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
            audio = sd.rec(sample_count,
                           samplerate=self.audio_samplerate,
                           channels=1,
                           blocking=True)
            return audio.flatten()
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
                        # Apply screen content factor to the camera brightness if available
                        if SCREEN_CAPTURE_AVAILABLE:
                            target_brightness = camera_brightness * self.screen_brightness_factor
                        else:
                            target_brightness = camera_brightness

                        # Check if change is significant enough
                        brightness_diff = abs(target_brightness - self.smoothed_brightness)
                        if brightness_diff > self.brightness_change_threshold:
                            # Smooth the transition
                            error = target_brightness - self.smoothed_brightness
                            self.smoothed_brightness += error * self.brightness_smoothing_factor
                            self.smoothed_brightness = max(
                                float(self.min_brightness),
                                min(float(self.max_brightness),
                                    self.smoothed_brightness))

                            # Apply the new brightness
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

                        # Map noise level to volume percentage
                        noise_range = self.max_noise_level - self.min_noise_level
                        normalized_noise = (noise_level - self.min_noise_level) / noise_range
                        normalized_noise = max(0.0, min(1.0, normalized_noise))
                        volume_range = self.max_volume - self.min_volume
                        target_volume = normalized_noise * volume_range + self.min_volume

                        # Smooth volume changes
                        volume_error = target_volume - self.smoothed_volume
                        self.smoothed_volume += volume_error * self.volume_smoothing_factor
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
            self.stop_event.set()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Check for audio dependencies if missing
    if not AUDIO_AVAILABLE:
        print("\nAudio features are disabled. To enable audio support, install PortAudio:")
        print("  sudo dnf install portaudio-devel  # For Fedora")
        print("  sudo apt install portaudio19-dev  # For Ubuntu/Debian")
        print("  Then reinstall the Python package: pip install sounddevice --user")
        print("\nContinuing without audio features...\n")
    
    # Check for PIL for screen capture
    if not SCREEN_CAPTURE_AVAILABLE:
        print("\nScreen content analysis is disabled. To enable, install one of:")
        print("  pip install mss --user        # Preferred method")
        print("  pip install pillow --user     # Alternative method")
        print("\nContinuing without screen content analysis...\n")
    
    try:
        controller = AdaptiveBrightnessVolumeController()
        print("Starting adaptive brightness and volume controller...")
        print("Detected brightness control method:", controller.brightness_method)
        if SCREEN_CAPTURE_AVAILABLE:
            print("Screen capture method:", SCREEN_CAPTURE_METHOD)
        print("Press Ctrl+C to stop")
        controller.run()
    except KeyboardInterrupt:
        print("\nStopping controller...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
