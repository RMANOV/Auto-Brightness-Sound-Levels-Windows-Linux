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

try:
    import mss  # type: ignore
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    print("Warning: mss not found. Screen content analysis disabled.")
    SCREEN_CAPTURE_AVAILABLE = False


class AdaptiveBrightnessVolumeController:
    def __init__(self, camera_index: int = 0,
                 brightness_range: Tuple[int, int] = (5, 45),
                 volume_range: Tuple[int, int] = (2, 60)):
        self.system = platform.system().lower()
        if self.system not in ["linux"]:
            print(f"Currently only Linux is supported. Detected: {self.system}")
            print("Support for Windows and other Linux distributions coming soon")
            sys.exit(1)
            
        # Check for required tools
        if os.system("which brightnessctl > /dev/null 2>&1") != 0:
            print("Error: brightnessctl not found! Please install it with:")
            print("  sudo apt install brightnessctl   # For Debian/Ubuntu")
            print("  sudo dnf install brightnessctl   # For Fedora")
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
        self.screen_check_interval: float = 1.0  # seconds
        self.last_screen_check_time: float = 0.0
        self.screen_brightness_factor: float = 1.0
        
        if SCREEN_CAPTURE_AVAILABLE:
            self.sct = mss.mss()
        else:
            self.sct = None

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
        self.cap = cv2.VideoCapture(self.camera_index)
        # Reduce camera resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_index}")
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
        if not SCREEN_CAPTURE_AVAILABLE or self.sct is None:
            return 1.0  # Neutral adjustment if screen capture not available

        try:
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)
            # Convert to numpy array and calculate brightness
            img = np.array(screenshot)
            brightness = np.mean(img) / 255

            # Adjust brightness factor based on screen content
            # For bright screens (white documents) reduce brightness
            # For dark screens (dark themes) increase brightness slightly
            if brightness > 0.7:  # Very bright content
                return 0.8  # Reduce screen brightness
            elif brightness < 0.3:  # Dark content
                return 1.2  # Increase screen brightness slightly
            else:
                return 1.0  # Neutral adjustment
        except Exception as e:
            print(f"Screen analysis error: {e}")
            return 1.0

    def get_brightness(self) -> float:
        # Use brightnessctl to get current brightness (Linux-specific)
        brightness = os.popen("brightnessctl get").read().strip()
        max_brightness = os.popen("brightnessctl max").read().strip()
        return float(brightness) / float(max_brightness) * 100

    def set_brightness(self, brightness: float) -> None:
        brightness = max(self.min_brightness, min(self.max_brightness, brightness))
        os.system(f"brightnessctl set {brightness}%")

    def get_volume(self) -> int:
        try:
            output = os.popen("amixer get Master").read()
            matches = re.search(r'\[([0-9]+)%\]', output)
            if matches:
                return int(matches.group(1))
            return 40
        except Exception:
            return 40

    def set_volume(self, volume: float) -> None:
        volume = max(self.min_volume, min(self.max_volume, int(volume)))
        os.system(f"amixer set Master {volume}%")

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
    controller = AdaptiveBrightnessVolumeController()
    print("Starting adaptive brightness and volume controller...")
    print("Press Ctrl+C to stop")
    controller.run()
