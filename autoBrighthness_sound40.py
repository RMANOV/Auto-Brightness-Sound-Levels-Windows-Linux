import cv2
import numpy as np
import os
import time
from threading import Thread, Event
from queue import Queue, Empty
from numba import njit
import sounddevice as sd
import re


class BrightnessController:
    def __init__(self, camera_index=0, frame_queue_size=50, brightness_queue_size=50,
                 frame_interval=0.1, update_interval=1, inactivity_threshold=300):
        self.camera_index = camera_index
        self.frame_queue_size = frame_queue_size
        self.brightness_queue_size = brightness_queue_size
        self.frame_interval = frame_interval
        self.update_interval = update_interval
        self.inactivity_threshold = inactivity_threshold
        self.setup_state()
        self.stop_event = Event()
        self.last_activity_time = time.time()
        self.is_active = True
        self.inactivity_check_interval = 1
        self.cap = cv2.VideoCapture(self.camera_index)
        # Reduce camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.prev_camera_brightness = None
        self.brightness_change_threshold = 10
        self.brightness_smoothing_factor = 0.5
        self.min_brightness = 10
        self.max_brightness = 95

        # Audio settings
        self.audio_duration = 0.1  # seconds
        self.audio_samplerate = 44100  # Hz
        self.min_volume = 2
        self.max_volume = 60
        self.prev_volume = self.get_volume()
        self.smoothed_volume = self.prev_volume
        self.volume_smoothing_factor = 0.95
        self.min_noise_level = 1e-5
        self.max_noise_level = 1e-2

    def setup_state(self):
        try:
            self.prev_brightness = self.get_brightness()
        except:
            self.prev_brightness = 50
        self.smoothed_brightness = self.prev_brightness

    def on_activity(self, *_):
        self.last_activity_time = time.time()
        self.is_active = True
        self.inactivity_check_interval = 1
        self.stop_event.clear()
        self.update_interval = max(self.update_interval / 2, 0.1)

    def on_inactivity(self):
        self.is_active = False
        self.stop_event.set()
        cv2.destroyAllWindows()
        self.update_interval = max(self.update_interval * 2, 1)

    @staticmethod
    @njit
    def calculate_brightness(frame):
        return frame.mean() / 255 * 100

    def analyze_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = self.calculate_brightness(frame)
        return brightness

    def process_frames(self, frame_queue, brightness_queue):
        while not self.stop_event.is_set():
            if self.is_active:
                try:
                    frame = frame_queue.get(timeout=1)
                    brightness = self.analyze_image(frame)
                    brightness_queue.put(brightness)
                except Empty:
                    continue
            else:
                time.sleep(self.inactivity_check_interval)

    def get_brightness(self):
        # Use brightnessctl to get current brightness
        brightness = os.popen("brightnessctl get").read().strip()
        max_brightness = os.popen("brightnessctl max").read().strip()
        return float(brightness) / float(max_brightness) * 100

    def set_brightness(self, brightness):
        brightness = max(self.min_brightness, min(
            self.max_brightness, brightness))

        # if brightness < 15:
        #     brightness = 15
        # if brightness >= 95:
        #     brightness = 90

        os.system(f"brightnessctl set {brightness}%")

    def get_volume(self):
        try:
            output = os.popen("amixer get Master").read()
            matches = re.search(r'\[([0-9]+)%\]', output)
            if matches:
                volume = int(matches.group(1))
                return volume
            else:
                return 50
        except:
            return 50

    def set_volume(self, volume):
        volume = max(self.min_volume, min(self.max_volume, volume))
        os.system(f"amixer set Master {volume}%")

    def capture_audio(self):
        audio = sd.rec(int(self.audio_duration * self.audio_samplerate),
                       samplerate=self.audio_samplerate, channels=1, blocking=True)
        return audio.flatten()

    def compute_noise_level(self, audio):
        rms = np.sqrt(np.mean(np.square(audio)))
        return rms

    def adjust_screen_brightness(self):
        frame_queue = Queue(maxsize=self.frame_queue_size)
        brightness_queue = Queue(maxsize=self.brightness_queue_size)

        process_thread = Thread(target=self.process_frames,
                                args=(frame_queue, brightness_queue))
        process_thread.daemon = True
        process_thread.start()

        last_brightness_change_time = time.time()

        try:
            while not self.stop_event.is_set():
                if time.time() - self.last_activity_time > self.inactivity_threshold:
                    self.on_inactivity()
                if self.is_active:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_queue.put(frame)
                        try:
                            camera_brightness = brightness_queue.get(
                                block=True, timeout=1)
                        except Empty:
                            camera_brightness = self.prev_camera_brightness if self.prev_camera_brightness is not None else 50
                    else:
                        camera_brightness = self.prev_camera_brightness if self.prev_camera_brightness is not None else 50

                    self.prev_camera_brightness = camera_brightness

                    brightness_diff = abs(
                        camera_brightness - self.smoothed_brightness)
                    if brightness_diff > self.brightness_change_threshold:
                        setpoint = camera_brightness
                    else:
                        setpoint = self.smoothed_brightness

                    error = setpoint - self.smoothed_brightness
                    self.smoothed_brightness += error * self.brightness_smoothing_factor
                    self.smoothed_brightness = max(self.min_brightness, min(
                        self.max_brightness, self.smoothed_brightness))

                    try:
                        self.set_brightness(round(self.smoothed_brightness))
                        self.prev_brightness = self.smoothed_brightness
                        last_brightness_change_time = time.time()
                    except:
                        self.smoothed_brightness = self.prev_brightness
                        self.update_interval = max(
                            self.update_interval * 2, 1)
                        exit()

                    # Adjust volume based on ambient noise
                    audio = self.capture_audio()
                    noise_level = self.compute_noise_level(audio)

                    # Map noise level to volume percentage
                    normalized_noise_level = (
                        noise_level - self.min_noise_level) / (self.max_noise_level - self.min_noise_level)
                    normalized_noise_level = max(
                        0.0, min(1.0, normalized_noise_level))
                    volume = normalized_noise_level * 100

                    # Smooth the volume changes
                    error_volume = volume - self.smoothed_volume
                    self.smoothed_volume += error_volume * self.volume_smoothing_factor
                    self.smoothed_volume = max(self.min_volume, min(
                        self.max_volume, self.smoothed_volume))

                    # Set the volume
                    self.set_volume(round(self.smoothed_volume))
                    self.prev_volume = self.smoothed_volume

                    self.update_interval = min(self.update_interval * 1.5, 5) if time.time(
                    ) - last_brightness_change_time > 5 else max(self.update_interval / 1.5, 0.1)
                    time.sleep(self.update_interval)
                else:
                    time.sleep(self.inactivity_check_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    controller = BrightnessController()
    controller.adjust_screen_brightness()
