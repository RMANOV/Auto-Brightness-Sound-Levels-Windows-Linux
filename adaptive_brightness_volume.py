import cv2
import numpy as np
import os
import time
import platform
from threading import Thread, Event, Lock
from queue import Queue, Empty
from numba import njit
import sounddevice as sd
import re
import mss
import sys


class AdaptiveBrightnessVolumeController:
    def __init__(self, camera_index=0, brightness_range=(5, 45), volume_range=(2, 60)):
        self.system = platform.system().lower()
        if self.system not in ["linux"]:
            print(f"Currently only Linux is supported. Detected: {self.system}")
            print("Support for Windows and other Linux distributions coming soon")
            sys.exit(1)
            
        # Configuration
        self.camera_index = camera_index
        self.min_brightness, self.max_brightness = brightness_range
        self.min_volume, self.max_volume = volume_range
        
        # Initialize state
        self.setup_camera()
        self.setup_state()
        
        # Threading and synchronization
        self.stop_event = Event()
        self.lock = Lock()
        
        # Activity tracking
        self.last_activity_time = time.time()
        self.is_active = True
        self.inactivity_threshold = 300  # seconds
        self.inactivity_check_interval = 1  # seconds
        
        # Smoothing parameters
        self.brightness_smoothing_factor = 0.3
        self.volume_smoothing_factor = 0.2
        self.brightness_change_threshold = 5
        
        # Screen content analysis
        self.sct = mss.mss()
        self.screen_check_interval = 1.0  # seconds
        self.last_screen_check_time = 0
        self.screen_brightness_factor = 1.0
        
        # Audio settings
        self.audio_duration = 0.1  # seconds
        self.audio_samplerate = 44100  # Hz
        self.min_noise_level = 1e-5
        self.max_noise_level = 1e-2

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        # Reduce camera resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_index}")
            print("Using fallback brightness control without ambient light detection")
            self.cap = None

    def setup_state(self):
        try:
            self.current_brightness = self.get_brightness()
        except:
            self.current_brightness = 30
        self.smoothed_brightness = self.current_brightness
        self.prev_camera_brightness = None
        
        try:
            self.current_volume = self.get_volume()
        except:
            self.current_volume = 40
        self.smoothed_volume = self.current_volume

    def on_activity(self):
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
            self.inactivity_check_interval = 1
            self.stop_event.clear()

    def on_inactivity(self):
        self.is_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    @staticmethod
    @njit
    def calculate_brightness(frame):
        return np.mean(frame) / 255 * 100

    def analyze_image(self, frame):
        if frame is None:
            return 50
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.calculate_brightness(gray_frame)

    def analyze_screen_content(self):
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

    def get_brightness(self):
        # Use brightnessctl to get current brightness (Linux-specific)
        brightness = os.popen("brightnessctl get").read().strip()
        max_brightness = os.popen("brightnessctl max").read().strip()
        return float(brightness) / float(max_brightness) * 100

    def set_brightness(self, brightness):
        brightness = max(self.min_brightness, min(self.max_brightness, brightness))
        os.system(f"brightnessctl set {brightness}%")

    def get_volume(self):
        try:
            output = os.popen("amixer get Master").read()
            matches = re.search(r'\[([0-9]+)%\]', output)
            if matches:
                return int(matches.group(1))
            return 40
        except:
            return 40

    def set_volume(self, volume):
        volume = max(self.min_volume, min(self.max_volume, volume))
        os.system(f"amixer set Master {volume}%")

    def capture_audio(self):
        try:
            audio = sd.rec(int(self.audio_duration * self.audio_samplerate),
                        samplerate=self.audio_samplerate, channels=1, blocking=True)
            return audio.flatten()
        except Exception as e:
            print(f"Audio capture error: {e}")
            return np.zeros(int(self.audio_duration * self.audio_samplerate))

    def compute_noise_level(self, audio):
        return np.sqrt(np.mean(np.square(audio)))

    def process_frames(self, frame_queue, brightness_queue):
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

    def run(self):
        frame_queue = Queue(maxsize=10)
        brightness_queue = Queue(maxsize=10)

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
                    if current_time - self.last_screen_check_time > self.screen_check_interval:
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
                        # Apply screen content factor to the camera brightness
                        target_brightness = camera_brightness * self.screen_brightness_factor
                        
                        # Check if change is significant enough
                        brightness_diff = abs(target_brightness - self.smoothed_brightness)
                        if brightness_diff > self.brightness_change_threshold:
                            # Smooth the transition
                            error = target_brightness - self.smoothed_brightness
                            self.smoothed_brightness += error * self.brightness_smoothing_factor
                            self.smoothed_brightness = max(self.min_brightness, 
                                                        min(self.max_brightness, 
                                                            self.smoothed_brightness))
                            
                            # Apply the new brightness
                            try:
                                self.set_brightness(round(self.smoothed_brightness))
                                self.current_brightness = self.smoothed_brightness
                                last_brightness_change_time = current_time
                            except Exception as e:
                                print(f"Brightness setting error: {e}")
                    
                    # Audio processing and volume adjustment
                    audio = self.capture_audio()
                    noise_level = self.compute_noise_level(audio)
                    
                    # Map noise level to volume percentage
                    normalized_noise = (noise_level - self.min_noise_level) / (self.max_noise_level - self.min_noise_level)
                    normalized_noise = max(0.0, min(1.0, normalized_noise))
                    target_volume = normalized_noise * (self.max_volume - self.min_volume) + self.min_volume
                    
                    # Smooth volume changes
                    volume_error = target_volume - self.smoothed_volume
                    self.smoothed_volume += volume_error * self.volume_smoothing_factor
                    self.smoothed_volume = max(self.min_volume, min(self.max_volume, self.smoothed_volume))
                    
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