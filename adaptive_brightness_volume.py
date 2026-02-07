#!/usr/bin/env python3
"""
Adaptive Brightness and Volume Controller
Patched: Windows support + NVIDIA virtual camera fix + Exposure lock

Changes from original (rust-rewrite branch):
  - __init__: Added lock_exposure param, Windows platform support
  - setup_camera: Rewritten with platform-aware backend selection
  - New: _setup_camera_windows, _setup_camera_linux, _apply_exposure_lock
  - New: _detect_brightness_method_windows, Windows get/set brightness/volume
"""

import cv2
import numpy as np
import os
import time
import platform
import sys
import signal
import gc
import atexit
import subprocess
from threading import Thread, Event, Lock
from queue import Queue, Empty
from typing import Optional, Tuple, cast
import re
import shutil
import functools
from collections import defaultdict

# Precompiled regex patterns for volume parsing
_RE_AMIXER_VOL = re.compile(r'\[([0-9]+)%\]')
_RE_PACTL_VOL = re.compile(r'(\d+)%')
_RE_WPCTL_VOL = re.compile(r'Volume: ([0-9.]+)')

# Performance monitoring
perf_timers = defaultdict(list)

def timeit(func_name):
    """Decorator to time function execution for performance monitoring"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            perf_timers[func_name].append(execution_time)
            if len(perf_timers[func_name]) > 100:
                perf_timers[func_name] = perf_timers[func_name][-100:]
            return result
        return wrapper
    return decorator

# Global cleanup registry for tracking resources
_cleanup_registry = []
_controller_instance = None

def register_cleanup(cleanup_func):
    """Register a cleanup function to be called on exit"""
    _cleanup_registry.append(cleanup_func)

def comprehensive_cleanup():
    """Comprehensive resource cleanup function"""
    print("Performing comprehensive resource cleanup...")

    global _controller_instance
    if _controller_instance:
        try:
            _controller_instance.stop_event.set()
            if hasattr(_controller_instance, 'process_thread') and _controller_instance.process_thread:
                if _controller_instance.process_thread.is_alive():
                    _controller_instance.process_thread.join(timeout=2.0)
        except Exception as e:
            print(f"Warning: Controller cleanup issue: {e}")

    try:
        cv2.destroyAllWindows()
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
            except Exception:
                break
    except Exception as e:
        print(f"Warning: OpenCV cleanup issue: {e}")

    try:
        import numba
        if hasattr(numba, 'cuda'):
            try:
                numba.cuda.close()
            except Exception:
                pass
        if hasattr(numba, 'typed'):
            numba.typed.List.empty_list.cache_clear()
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Numba cleanup issue: {e}")

    global perf_timers
    perf_timers.clear()

    for cleanup_func in _cleanup_registry:
        try:
            cleanup_func()
        except Exception as e:
            print(f"Warning: Cleanup function failed: {e}")

    for _ in range(3):
        collected = gc.collect()
        if collected == 0:
            break

    print(f"Cleanup completed - collected {gc.collect()} objects")

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print(f"\nReceived signal {signum}, initiating cleanup...")
    comprehensive_cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(comprehensive_cleanup)

# Try to use Rust backend for maximum performance
USE_RUST = False
try:
    import adaptive_rust
    USE_RUST = True
    print(f"Using Rust backend v{adaptive_rust.version()} (2-4x faster than Numba)")
except ImportError:
    pass

try:
    from numba import njit  # type: ignore
except ImportError:
    if not USE_RUST:
        print("Warning: numba not found. Using fallback implementation.")
    def njit(func):
        return func

# Audio detection
AUDIO_AVAILABLE = False
AUDIO_METHOD = ""

try:
    import sounddevice as sd  # type: ignore
    AUDIO_AVAILABLE = True
    AUDIO_METHOD = "sounddevice"
except (ImportError, OSError) as e:
    print(f"Warning: sounddevice not available - {e}")

if not AUDIO_AVAILABLE and platform.system().lower() == "linux":
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

# Screen capture
SCREEN_CAPTURE_METHOD = ""
SCREEN_CAPTURE_AVAILABLE = False

try:
    from PIL import ImageGrab  # type: ignore
    try:
        test_grab = ImageGrab.grab(bbox=(0, 0, 10, 10))
        test_grab.size
        SCREEN_CAPTURE_METHOD = "pillow"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found PIL.ImageGrab for screen content analysis")
    except Exception as e:
        print(f"Warning: PIL.ImageGrab is installed but not functional: {e}")
except ImportError:
    print("Warning: PIL.ImageGrab not found. Trying alternative methods.")

if not SCREEN_CAPTURE_AVAILABLE:
    try:
        import mss  # type: ignore
        SCREEN_CAPTURE_METHOD = "mss"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found MSS for screen content analysis")
    except ImportError:
        print("Warning: MSS not found.")

if not SCREEN_CAPTURE_AVAILABLE and platform.system().lower() == "linux":
    has_xrandr = os.system("which xrandr > /dev/null 2>&1") == 0
    has_import = os.system("which import > /dev/null 2>&1") == 0

    if has_xrandr and has_import:
        SCREEN_CAPTURE_METHOD = "xrandr-import"
        SCREEN_CAPTURE_AVAILABLE = True
        print("Found xrandr/import tools for screen content analysis")
        os.makedirs(os.path.expanduser("~/.cache/adaptive-controller"), exist_ok=True)

if not SCREEN_CAPTURE_AVAILABLE and platform.system().lower() == "linux":
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
    """
    High-Performance Adaptive Brightness and Volume Controller

    Supports Linux and Windows. Uses camera-based ambient light detection
    with exposure lock for reliable measurements.
    """
    def __init__(self, camera_index: int = 0,
                 lock_exposure: bool = True,
                 brightness_range: Tuple[int, int] = (5, 45),
                 volume_range: Tuple[int, int] = (3, 35),
                 auto_exit: bool = True):
        self.system = platform.system().lower()
        if self.system not in ["linux", "windows"]:
            print(f"Currently only Linux and Windows are supported. Detected: {self.system}")
            sys.exit(1)

        # Check for brightness control tools
        if self.system == "linux":
            self.brightness_method = self._detect_brightness_method()
        else:
            self.brightness_method = self._detect_brightness_method_windows()

        if not self.brightness_method:
            print("Error: No supported brightness control method found!")
            if self.system == "linux":
                print("Options:")
                print("  1. Install brightnessctl: sudo dnf install brightnessctl")
                print("  2. Install xbacklight: sudo dnf install xbacklight")
                print("  3. Make sure /sys/class/backlight/ is accessible")
            else:
                print("Windows WMI brightness control not available.")
                print("Make sure you're on a laptop with WMI brightness support.")
            sys.exit(1)

        # Cache volume control tool (detect once, not every call)
        self.volume_tool: Optional[str] = None
        if self.system == "linux":
            for tool in ("amixer", "pactl", "wpctl"):
                if shutil.which(tool):
                    self.volume_tool = tool
                    break

        # Auto-exit: stop once brightness & volume converge
        self.auto_exit: bool = auto_exit

        # Configuration
        self.camera_index: int = camera_index
        self.lock_exposure: bool = lock_exposure
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
        self.process_thread: Optional[Thread] = None
        self.frame_queue: Optional[Queue] = None
        self.brightness_queue: Optional[Queue] = None

        # Activity tracking
        self.last_activity_time: float = time.time()
        self.is_active: bool = True
        self.inactivity_threshold: int = 300
        self.inactivity_check_interval: float = 1.0

        global _controller_instance
        _controller_instance = self
        register_cleanup(self._instance_cleanup)

    def _instance_cleanup(self):
        """Instance-specific cleanup method"""
        try:
            self.stop_event.set()
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=2.0)
                if self.process_thread.is_alive():
                    print("Warning: Process thread did not terminate cleanly")
            if self.frame_queue:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
            if self.brightness_queue:
                while not self.brightness_queue.empty():
                    try:
                        self.brightness_queue.get_nowait()
                    except Empty:
                        break
            if self.cap and self.cap.isOpened():
                self.cap.release()
            print("Instance cleanup completed")
        except Exception as e:
            print(f"Warning: Instance cleanup error: {e}")

    # ========================================================================
    # CAMERA SETUP - Platform-aware with NVIDIA virtual camera workaround
    # ========================================================================

    def setup_camera(self):
        """Initialize camera with platform-aware backend selection.

        On Windows, uses DirectShow (DSHOW) backend to enable exposure lock
        and skips NVIDIA Broadcast virtual cameras that steal index 0.
        On Linux, uses the default V4L2 backend.
        """
        self.cap = None
        self._camera_backend = cv2.CAP_ANY

        if self.system == "windows":
            self.cap = self._setup_camera_windows()
        else:
            self.cap = self._setup_camera_linux()

        # Configure resolution and exposure if camera found
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            print(f"Camera initialized at index {self.camera_index} "
                  f"(backend: {self.cap.getBackendName()})")

            if self.lock_exposure:
                self._apply_exposure_lock()
        else:
            print("No working camera found")
            print("Using fallback brightness control without ambient sensing")
            self.cap = cast(Optional[cv2.VideoCapture], None)

    def _setup_camera_windows(self) -> Optional[cv2.VideoCapture]:
        """Windows-specific camera setup using DSHOW backend.

        NVIDIA Broadcast installs a virtual camera (VCAMDS) that occupies
        index 0 in DirectShow and causes C++ exceptions. This method:
        1. Tries DSHOW backend on each index
        2. Validates the camera supports exposure control (real cameras do,
           virtual cameras typically don't)
        3. Falls back to MSMF if DSHOW fails entirely
        """
        # Phase 1: Try DSHOW backend (supports exposure control)
        for idx in range(10):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Test if this is a real camera by checking exposure control
                    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                    can_set = cap.set(cv2.CAP_PROP_EXPOSURE, -5)
                    readback = cap.get(cv2.CAP_PROP_EXPOSURE)

                    if exposure != -1.0 or (can_set and readback != -1.0):
                        # Real camera: exposure is readable/settable
                        self.camera_index = idx
                        self._camera_backend = cv2.CAP_DSHOW
                        print(f"Found real camera at DSHOW index {idx} "
                              f"(exposure={exposure}, readback={readback})")
                        return cap
                    else:
                        print(f"Skipping virtual camera at DSHOW index {idx}")
                        cap.release()
                else:
                    cap.release()
            except Exception:
                # NVIDIA virtual cam can throw C++ exceptions at index 0
                continue

        # Phase 2: Fallback to MSMF (no exposure lock, but still works)
        print("DSHOW failed for all indices, trying MSMF backend...")
        for idx in range(10):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.camera_index = idx
                        self._camera_backend = cv2.CAP_MSMF
                        self.lock_exposure = False
                        print(f"Using MSMF fallback at index {idx} "
                              f"(exposure lock not available)")
                        return cap
                cap.release()
            except Exception:
                continue

        return None

    def _setup_camera_linux(self) -> Optional[cv2.VideoCapture]:
        """Linux camera setup (original behavior preserved)."""
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_index}, "
                  f"trying alternatives...")
            for idx in range(10):
                if idx == self.camera_index:
                    continue
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    print(f"Found working camera at index {idx}")
                    self.camera_index = idx
                    return test_cap
                else:
                    test_cap.release()
            return None

        return cap

    def _apply_exposure_lock(self):
        """Lock camera exposure for reliable ambient light measurement.

        With auto-exposure (default), the camera compensates for light changes,
        making brightness readings nearly constant regardless of actual room
        lighting. Locking exposure means np.mean(frame) directly correlates
        with real ambient light.

        Exposure -4 gives good dynamic range:
          Dark room:   brightness ~20
          Bright room: brightness ~80+
        """
        if self.cap is None or not self.cap.isOpened():
            return

        # Disable auto-exposure (1 = manual mode for most cameras)
        ok_ae = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        # Fixed exposure for good ambient light dynamic range
        ok_ex = self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)

        if ok_ae or ok_ex:
            readback = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"Exposure locked at {readback:.0f} "
                  f"(set_auto_exp={ok_ae}, set_exp={ok_ex})")
            # Flush frames to let new exposure settle
            for _ in range(10):
                self.cap.read()
        else:
            print("Warning: Exposure lock not supported by this camera/backend")
            print("Ambient light readings may be unreliable due to auto-exposure")
            self.lock_exposure = False

    # ========================================================================
    # BRIGHTNESS CONTROL - Platform detection and get/set
    # ========================================================================

    def _detect_brightness_method(self) -> str:
        """Detect available brightness control method (Linux)"""
        if os.system("which brightnessctl > /dev/null 2>&1") == 0:
            return "brightnessctl"
        if os.system("which xbacklight > /dev/null 2>&1") == 0:
            return "xbacklight"
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

    def _detect_brightness_method_windows(self) -> str:
        """Detect brightness control method on Windows via WMI."""
        try:
            result = subprocess.run(
                ['powershell.exe', '-Command',
                 'Get-CimInstance -Namespace root/WMI '
                 '-ClassName WmiMonitorBrightness '
                 '| Select-Object -ExpandProperty CurrentBrightness'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return "wmi"
        except Exception:
            pass
        return ""

    def get_brightness(self) -> float:
        """Get current screen brightness as percentage"""
        if self.system == "windows" and self.brightness_method == "wmi":
            try:
                result = subprocess.run(
                    ['powershell.exe', '-Command',
                     '(Get-CimInstance -Namespace root/WMI '
                     '-ClassName WmiMonitorBrightness).CurrentBrightness'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except Exception as e:
                print(f"Error reading Windows brightness: {e}")
            return 50.0

        if self.brightness_method == "brightnessctl":
            try:
                cur = subprocess.run(["brightnessctl", "get"], capture_output=True, text=True, timeout=3)
                mx = subprocess.run(["brightnessctl", "max"], capture_output=True, text=True, timeout=3)
                return float(cur.stdout.strip()) / float(mx.stdout.strip()) * 100
            except Exception as e:
                print(f"Error reading brightnessctl: {e}")
                return 50.0
        elif self.brightness_method == "xbacklight":
            try:
                r = subprocess.run(["xbacklight", "-get"], capture_output=True, text=True, timeout=3)
                return float(r.stdout.strip())
            except Exception as e:
                print(f"Error reading xbacklight: {e}")
                return 50.0
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
        return 50.0

    def set_brightness(self, brightness: float) -> None:
        """Set screen brightness as percentage"""
        brightness = max(self.min_brightness, min(self.max_brightness, brightness))
        calibrated_brightness = round(brightness)
        calibrated_brightness = max(self.min_brightness,
                                    min(self.max_brightness, calibrated_brightness))
        print(f"Brightness: Target: {brightness:.1f}% -> Setting: {calibrated_brightness}%")

        if self.system == "windows" and self.brightness_method == "wmi":
            try:
                subprocess.run(
                    ['powershell.exe', '-Command',
                     f'(Get-WmiObject -Namespace root/WMI '
                     f'-Class WmiMonitorBrightnessMethods)'
                     f'.WmiSetBrightness(1,{calibrated_brightness})'],
                    capture_output=True, timeout=5
                )
            except Exception as e:
                print(f"Error setting Windows brightness: {e}")
            return

        if self.brightness_method == "brightnessctl":
            success = False
            result1 = os.system(f"brightnessctl set {calibrated_brightness}% >/dev/null 2>&1")
            if result1 == 0:
                success = True
            if not success:
                result2 = os.system(f"/usr/bin/brightnessctl set {calibrated_brightness}% >/dev/null 2>&1")
                if result2 == 0:
                    success = True
            if not success:
                result3 = os.system(f"sudo -n /usr/bin/brightnessctl set {calibrated_brightness}% >/dev/null 2>&1")
                if result3 == 0:
                    success = True
            if not success:
                try:
                    backlight_dir = "/sys/class/backlight/intel_backlight"
                    if os.path.exists(backlight_dir):
                        with open(f"{backlight_dir}/max_brightness", "r") as f:
                            max_brightness = int(f.read().strip())
                        value = int((calibrated_brightness / 100) * max_brightness)
                        try:
                            with open(f"{backlight_dir}/brightness", "w") as f:
                                f.write(str(value))
                            success = True
                        except PermissionError:
                            result4 = os.system(f"echo {value} | sudo -n tee {backlight_dir}/brightness >/dev/null 2>&1")
                            if result4 == 0:
                                success = True
                except Exception as e:
                    print(f"DEBUG: Sysfs fallback failed: {e}")
            if not success:
                print(f"Warning: Failed to set brightness - all methods failed")
        elif self.brightness_method == "xbacklight":
            os.system(f"xbacklight -set {calibrated_brightness}")
        elif self.brightness_method == "sysfs":
            try:
                with open(f"{self.backlight_dir}/max_brightness", "r") as f:
                    max_brightness = int(f.read().strip())
                value = int((calibrated_brightness / 100) * max_brightness)
                try:
                    with open(f"{self.backlight_dir}/brightness", "w") as f:
                        f.write(str(value))
                except PermissionError:
                    os.system(f"echo {value} | sudo tee {self.backlight_dir}/brightness > /dev/null")
            except Exception as e:
                print(f"Error setting brightness: {e}")

    # ========================================================================
    # VOLUME CONTROL
    # ========================================================================

    def get_volume(self) -> int:
        """Get current volume level as percentage"""
        if self.system == "windows":
            try:
                # Use PowerShell with Windows Audio Session API
                result = subprocess.run(
                    ['powershell.exe', '-Command',
                     '[Math]::Round('
                     '(New-Object -ComObject WScript.Shell)'
                     '.RegRead("HKCU\\SOFTWARE\\Microsoft\\Multimedia'
                     '\\Audio\\Volume") / 65535 * 100)'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    return int(result.stdout.strip())
            except Exception:
                pass
            return 40

        try:
            if self.volume_tool == "amixer":
                result = subprocess.run(["amixer", "get", "Master"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    m = _RE_AMIXER_VOL.search(result.stdout)
                    if m:
                        return int(m.group(1))
            elif self.volume_tool == "pactl":
                result = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    m = _RE_PACTL_VOL.search(result.stdout)
                    if m:
                        return int(m.group(1))
            elif self.volume_tool == "wpctl":
                result = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    m = _RE_WPCTL_VOL.search(result.stdout)
                    if m:
                        return int(float(m.group(1)) * 100)
            return 40
        except Exception as e:
            print(f"Error getting volume: {e}")
            return 40

    def set_volume(self, volume: float) -> None:
        """Set volume level as percentage"""
        volume = max(self.min_volume, min(self.max_volume, int(volume)))

        if self.system == "windows":
            try:
                # Use nircmd if available, otherwise PowerShell
                # nircmd is the most reliable way to set exact volume on Windows
                nircmd_path = os.path.expanduser("~/.local/bin/nircmd.exe")
                if os.path.exists(nircmd_path):
                    vol_value = int(volume / 100 * 65535)
                    subprocess.run(
                        [nircmd_path, 'setsysvolume', str(vol_value)],
                        capture_output=True, timeout=5
                    )
                else:
                    # PowerShell fallback using AudioDeviceCmdlets or WScript
                    subprocess.run(
                        ['powershell.exe', '-Command',
                         f'$obj = New-Object -ComObject WScript.Shell; '
                         f'1..50 | ForEach-Object {{ $obj.SendKeys([char]174) }}; '
                         f'1..{max(1, volume // 2)} | ForEach-Object {{ $obj.SendKeys([char]175) }}'],
                        capture_output=True, timeout=10
                    )
            except Exception as e:
                print(f"Warning: Failed to set volume on Windows: {e}")
            return

        success = False
        vol = int(volume)
        try:
            if self.volume_tool == "amixer":
                r = subprocess.run(["amixer", "set", "Master", f"{vol}%"], capture_output=True, timeout=3)
                success = (r.returncode == 0)
            elif self.volume_tool == "pactl":
                r = subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"], capture_output=True, timeout=3)
                success = (r.returncode == 0)
            elif self.volume_tool == "wpctl":
                r = subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{vol / 100.0:.2f}"], capture_output=True, timeout=3)
                success = (r.returncode == 0)
        except Exception as e:
            print(f"Warning: Volume set error: {e}")
        if not success:
            print(f"Warning: Failed to set volume to {vol}%")

    # ========================================================================
    # STATE & SMOOTHING
    # ========================================================================

    def load_saved_state(self) -> Tuple[Optional[float], Optional[float]]:
        """Load previously saved brightness and volume settings"""
        try:
            config_file = os.path.expanduser("~/.config/adaptive-controller/last_state.txt")
            if os.path.exists(config_file):
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
                if timestamp and time.time() - timestamp < 24 * 60 * 60:
                    return brightness, volume
        except Exception as e:
            print(f"Warning: Could not load saved settings: {e}")
        return None, None

    def setup_state(self) -> None:
        self.brightness_smoothing_factor: float = 0.3
        self.volume_smoothing_factor: float = 0.2
        self.brightness_change_threshold: float = 5.0

        self.screen_check_interval: float = 2.0
        self.last_screen_check_time: float = 0.0
        self.screen_brightness_factor: float = 1.0
        self.screen_capture_error_count: int = 0
        self.screen_capture_enabled: bool = True
        self.max_screen_errors: int = 5

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

        self.audio_duration: float = 0.1
        self.audio_samplerate: int = 44100
        self.min_noise_level: float = 5e-6
        self.max_noise_level: float = 8e-3

        self.warmup_frames: int = 20
        self.current_warmup_frame: int = 0
        self.warmup_cooldown: float = 0.05
        self.is_in_warmup: bool = True
        self.initial_brightness: Optional[float] = None
        self.initial_volume: Optional[float] = None

        self.audio_warmup_frames: int = 40
        self.audio_warmup_cooldown: float = 0.005
        self.is_audio_in_warmup: bool = True
        self.audio_warmup_threshold: int = 10

        self.brightness_calibration_factor: float = 0.4
        self.last_significant_change_time: float = 0.0
        self.sensitivity_to_changes: float = 1.5

        saved_brightness, saved_volume = self.load_saved_state()

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

    # ========================================================================
    # COMPUTATION - JIT/Rust accelerated
    # ========================================================================

    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        if USE_RUST:
            return adaptive_rust.calculate_brightness(frame.flatten())
        return np.mean(frame) / 255.0 * 100.0

    @staticmethod
    def _calculate_brightness_mapping_jit(camera_brightness: float,
                                          min_brightness: float,
                                          max_brightness: float) -> float:
        if USE_RUST:
            return adaptive_rust.calculate_brightness_mapping(
                camera_brightness, min_brightness, max_brightness)
        base_linear = min_brightness + (camera_brightness * 0.35)
        boost_factor = 1.0
        if 35.0 <= camera_brightness <= 55.0:
            distance_from_45 = abs(camera_brightness - 45.0)
            max_boost = 1.35
            boost_factor = max_boost - (distance_from_45 / 10.0 * (max_boost - 1.0))
        target_brightness = base_linear * boost_factor
        return max(min_brightness, min(max_brightness, target_brightness))

    @staticmethod
    def _calculate_volume_mapping_jit(normalized_noise: float,
                                      min_volume: float,
                                      max_volume: float) -> float:
        if USE_RUST:
            return adaptive_rust.calculate_volume_mapping(
                normalized_noise, min_volume, max_volume)
        if normalized_noise > 0.0:
            curve_factor = 0.55
            multiplier = 12.0
            bias = 0.22
            normalized_noise_enhanced = min(normalized_noise**0.8 * 1.2, 1.0)
            adjusted_noise = curve_factor * np.log10(1.0 + multiplier * normalized_noise_enhanced) + bias
            adjusted_noise = max(0.0, min(1.0, adjusted_noise))
        else:
            adjusted_noise = 0.22
        volume_range = max_volume - min_volume
        return adjusted_noise * volume_range + min_volume

    @staticmethod
    def _smooth_transition_jit(current_value: float,
                               target_value: float,
                               smoothing_factor: float) -> float:
        if USE_RUST:
            return adaptive_rust.smooth_transition(
                current_value, target_value, smoothing_factor)
        error = target_value - current_value
        return current_value + error * smoothing_factor

    @staticmethod
    def _analyze_screen_brightness_jit(img_array: np.ndarray) -> float:
        if USE_RUST:
            return adaptive_rust.analyze_screen_brightness(img_array.flatten().astype(np.uint8))
        brightness = np.mean(img_array) / 255.0
        if brightness > 0.7:
            return 1.2
        elif brightness < 0.3:
            return 0.8
        else:
            return 1.0

    @staticmethod
    def _check_significant_change_jit(current_brightness: float,
                                      last_brightness: float,
                                      is_dimming: bool) -> bool:
        if USE_RUST:
            return adaptive_rust.check_significant_change(
                current_brightness, last_brightness, is_dimming)
        brightness_change = current_brightness - last_brightness
        abs_change = abs(brightness_change)
        dimming_threshold = 8.0
        brightening_threshold = 12.0
        if is_dimming and brightness_change < 0.0 and abs_change > dimming_threshold:
            return True
        elif not is_dimming and brightness_change > 0.0 and abs_change > brightening_threshold:
            return True
        else:
            return False

    def print_performance_stats(self) -> None:
        if not perf_timers:
            return
        print("\nPerformance Statistics:")
        print("=" * 60)
        total_calls = 0
        total_time = 0
        for func_name, times in perf_timers.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                calls = len(times)
                total_calls += calls
                total_time += sum(times)
                print(f"{func_name:20s}: {avg_time:6.2f}ms avg ({min_time:5.2f}-{max_time:5.2f}ms) [{calls:3d} calls]")
        if total_calls > 0:
            print(f"{'TOTAL':20s}: {total_time:6.1f}ms total, {total_calls:3d} calls")
            print(f"{'EFFICIENCY':20s}: {total_time/total_calls:6.2f}ms per operation")
        print("=" * 60)

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    @timeit("analyze_image")
    def analyze_image(self, frame: Optional[np.ndarray]) -> float:
        if frame is None:
            return 50.0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.calculate_brightness(gray_frame)

    @timeit("analyze_screen_content")
    def analyze_screen_content(self) -> float:
        if not SCREEN_CAPTURE_AVAILABLE or not self.screen_capture_enabled:
            return 1.0
        try:
            if self.pil_available:
                try:
                    screenshot = ImageGrab.grab()
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"PIL capture failed: {e}")
            elif self.xrandr_available:
                try:
                    process = subprocess.run(
                        ["xrandr", "--current"],
                        capture_output=True, text=True, check=True
                    )
                    output = process.stdout
                    lines = output.strip().split('\n')
                    resolution = None
                    for line in lines:
                        if "*" in line and "+" in line:
                            parts = line.split()
                            for part in parts:
                                if "x" in part and part[0].isdigit():
                                    resolution = part
                                    break
                            if resolution:
                                break
                    if not resolution:
                        resolution = "1920x1080"
                    width, height = map(int, resolution.split("x"))
                    center_x = width // 4
                    center_y = height // 4
                    center_width = width // 2
                    center_height = height // 2
                    cmd = f"import -window root -crop {center_width}x{center_height}+{center_x}+{center_y} {self.screenshot_path}"
                    result = os.system(cmd)
                    if result != 0:
                        raise Exception("Failed to capture screenshot with import")
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
                    monitor = self.sct.monitors[1]
                    screenshot = self.sct.grab(monitor)
                    img = np.array(screenshot)
                except Exception as e:
                    raise Exception(f"MSS capture failed: {e}")
            else:
                return 1.0

            brightness_factor = self._analyze_screen_brightness_jit(img)
            self.screen_capture_error_count = 0
            return brightness_factor
        except Exception as e:
            if self.screen_capture_error_count < self.max_screen_errors:
                print(f"Screen analysis error: {e}")
                self.screen_capture_error_count += 1
            elif self.screen_capture_error_count == self.max_screen_errors:
                print("Too many screen capture errors, suppressing further messages")
                self.screen_capture_error_count += 1
            if self.screen_capture_error_count > self.max_screen_errors + 10:
                print(f"Warning: Disabling problematic screen capture method: {SCREEN_CAPTURE_METHOD}")
                self.screen_capture_enabled = False
            return 1.0

    @timeit("capture_audio")
    def capture_audio(self) -> np.ndarray:
        if not AUDIO_AVAILABLE:
            return np.zeros(int(self.audio_duration * self.audio_samplerate))
        try:
            sample_count = int(self.audio_duration * self.audio_samplerate)
            if AUDIO_METHOD == "sounddevice":
                audio = sd.rec(sample_count,
                               samplerate=self.audio_samplerate,
                               channels=1,
                               blocking=True)
                return audio.flatten()
            elif AUDIO_METHOD == "arecord":
                audio_file = os.path.expanduser("~/.cache/adaptive-controller/audio.wav")
                sample_count = int(self.audio_duration * self.audio_samplerate)
                cmd = f"arecord -q --samples={sample_count} -f S16_LE -r {self.audio_samplerate} -c1 {audio_file}"
                result = os.system(cmd)
                if result != 0:
                    raise Exception("Failed to record audio with arecord")
                import wave
                with wave.open(audio_file, 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
            elif AUDIO_METHOD == "sox":
                audio_file = os.path.expanduser("~/.cache/adaptive-controller/audio.wav")
                cmd = f"sox -n -r {self.audio_samplerate} -c 1 {audio_file} trim 0 {self.audio_duration}"
                result = os.system(cmd)
                if result != 0:
                    raise Exception("Failed to record audio with sox")
                import wave
                with wave.open(audio_file, 'rb') as wf:
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
            return np.zeros(int(self.audio_duration * self.audio_samplerate))
        except Exception as e:
            print(f"Audio capture error: {e}")
            return np.zeros(int(self.audio_duration * self.audio_samplerate))

    @timeit("compute_noise_level")
    def compute_noise_level(self, audio: np.ndarray) -> float:
        return self._compute_noise_level_jit(audio)

    @staticmethod
    def _compute_noise_level_jit(audio: np.ndarray) -> float:
        if USE_RUST:
            return adaptive_rust.compute_noise_level(audio.astype(np.float32))
        return np.sqrt(np.mean(np.square(audio)))

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

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
        start_time = time.time()
        self.frame_queue = Queue(maxsize=10)
        self.brightness_queue = Queue(maxsize=10)

        if self.cap and self.cap.isOpened():
            self.process_thread = Thread(target=self.process_frames, args=(self.frame_queue, self.brightness_queue))
            self.process_thread.daemon = True
            self.process_thread.start()

        update_interval = 0.5
        last_brightness_change_time = time.time()
        last_perf_report_time = time.time()
        perf_report_interval = 30.0

        # Auto-exit convergence tracking
        _converge_count = 0
        _converge_threshold = 1.0  # % delta to consider "stable"
        _converge_required = 3     # consecutive stable frames to confirm
        _last_target_brightness = None
        _last_target_volume = None

        try:
            while not self.stop_event.is_set():
                current_time = time.time()

                if current_time - self.last_activity_time > self.inactivity_threshold:
                    self.on_inactivity()

                if current_time - last_perf_report_time > perf_report_interval:
                    self.print_performance_stats()
                    last_perf_report_time = current_time

                if self.is_active:
                    if SCREEN_CAPTURE_AVAILABLE and current_time - self.last_screen_check_time > self.screen_check_interval:
                        self.screen_brightness_factor = self.analyze_screen_content()
                        self.last_screen_check_time = current_time

                    camera_brightness = None
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            self.frame_queue.put(frame)
                            try:
                                camera_brightness = self.brightness_queue.get(block=False)
                                self.prev_camera_brightness = camera_brightness
                            except Empty:
                                camera_brightness = self.prev_camera_brightness
                    else:
                        camera_brightness = self.prev_camera_brightness

                    if camera_brightness is not None:
                        last_camera_brightness = self.prev_camera_brightness
                        if last_camera_brightness is not None:
                            brightness_change = camera_brightness - last_camera_brightness
                            is_dimming = brightness_change < 0
                            if self._check_significant_change_jit(camera_brightness, last_camera_brightness, is_dimming):
                                self.last_significant_change_time = current_time
                                direction = "DIMMING" if is_dimming else "BRIGHTENING"
                                print(f"Light change detected - {direction}: "
                                      f"{last_camera_brightness:.1f} -> {camera_brightness:.1f} "
                                      f"(delta {brightness_change:.1f})")
                                if is_dimming:
                                    self.sensitivity_to_changes = 2.5
                                else:
                                    self.sensitivity_to_changes = 1.5

                        if self.current_warmup_frame % 10 == 0 or current_time - self.last_significant_change_time < 5:
                            print(f"Camera brightness: {camera_brightness:.1f}")

                        base_adjustment_speed = 1.5
                        time_since_change = current_time - self.last_significant_change_time
                        if time_since_change < 5:
                            adjustment_boost = 2.0 * base_adjustment_speed
                        else:
                            adjustment_boost = base_adjustment_speed

                        target_brightness = self._calculate_brightness_mapping_jit(
                            camera_brightness,
                            float(self.min_brightness),
                            float(self.max_brightness)
                        )

                        if SCREEN_CAPTURE_AVAILABLE:
                            target_brightness = target_brightness * self.screen_brightness_factor

                        if self.is_in_warmup or self.is_audio_in_warmup:
                            self.current_warmup_frame += 1
                        if self.is_in_warmup:
                            if self.initial_brightness is None:
                                try:
                                    self.initial_brightness = self.get_brightness()
                                    self.smoothed_brightness = self.initial_brightness
                                    print(f"Starting from current brightness: {self.initial_brightness}%")
                                except Exception:
                                    self.initial_brightness = self.smoothed_brightness
                            if self.current_warmup_frame <= self.warmup_frames:
                                if self.current_warmup_frame % 4 == 0:
                                    print(f"Calibrating... {(self.current_warmup_frame * 100) // self.warmup_frames}%")
                                self.smoothed_brightness = self._smooth_transition_jit(
                                    self.smoothed_brightness,
                                    target_brightness,
                                    self.brightness_smoothing_factor * self.warmup_cooldown
                                )
                            else:
                                self.is_in_warmup = False
                                print("Calibration complete, applying normal brightness control")
                                self.prev_camera_brightness = camera_brightness
                        else:
                            smooth_factor = self.brightness_smoothing_factor * adjustment_boost
                            old_brightness = self.smoothed_brightness
                            self.smoothed_brightness = self._smooth_transition_jit(
                                self.smoothed_brightness,
                                target_brightness,
                                smooth_factor
                            )
                            error = target_brightness - old_brightness
                            if abs(error) > 2.0:
                                print(f"Adjusting brightness: {old_brightness:.1f}% -> "
                                      f"{target_brightness:.1f}% "
                                      f"(rate: {smooth_factor:.2f}, step: {error * smooth_factor:.2f})")

                        self.smoothed_brightness = max(
                            float(self.min_brightness),
                            min(float(self.max_brightness),
                                self.smoothed_brightness))

                        try:
                            self.set_brightness(round(self.smoothed_brightness))
                            self.current_brightness = self.smoothed_brightness
                            last_brightness_change_time = current_time
                        except Exception as e:
                            print(f"Brightness setting error: {e}")

                    if AUDIO_AVAILABLE:
                        audio = self.capture_audio()
                        noise_level = self.compute_noise_level(audio)
                        noise_range = self.max_noise_level - self.min_noise_level
                        normalized_noise = (noise_level - self.min_noise_level) / noise_range
                        normalized_noise = max(0.0, min(1.0, normalized_noise))
                        target_volume = self._calculate_volume_mapping_jit(
                            normalized_noise,
                            float(self.min_volume),
                            float(self.max_volume)
                        )

                        if self.is_audio_in_warmup:
                            if self.initial_volume is None:
                                try:
                                    current_system_volume = float(self.get_volume())
                                    self.initial_volume = current_system_volume
                                    self.smoothed_volume = current_system_volume
                                    print(f"Initial volume locked at: {self.initial_volume}%")
                                except Exception:
                                    self.initial_volume = self.smoothed_volume
                            if self.current_warmup_frame <= self.audio_warmup_threshold:
                                self.smoothed_volume = self.initial_volume
                                volume_change = 0
                            elif self.current_warmup_frame <= self.audio_warmup_frames:
                                if self.current_warmup_frame % 8 == 0:
                                    audio_progress = ((self.current_warmup_frame - self.audio_warmup_threshold) * 100) // (self.audio_warmup_frames - self.audio_warmup_threshold)
                                    print(f"Audio calibrating... {audio_progress}%")
                                self.smoothed_volume = self._smooth_transition_jit(
                                    self.smoothed_volume,
                                    target_volume,
                                    self.volume_smoothing_factor * self.audio_warmup_cooldown
                                )
                                volume_change = self.smoothed_volume - target_volume
                            else:
                                self.is_audio_in_warmup = False
                                print("Audio calibration complete")
                                old_volume = self.smoothed_volume
                                self.smoothed_volume = self._smooth_transition_jit(
                                    self.smoothed_volume,
                                    target_volume,
                                    self.volume_smoothing_factor * 0.5
                                )
                                volume_change = self.smoothed_volume - old_volume
                        else:
                            old_volume = self.smoothed_volume
                            self.smoothed_volume = self._smooth_transition_jit(
                                self.smoothed_volume,
                                target_volume,
                                self.volume_smoothing_factor
                            )
                            volume_change = self.smoothed_volume - old_volume

                        if abs(volume_change) > 0.1:
                            print(f"Volume: {self.smoothed_volume:.1f}% (change: {volume_change:.2f})")

                        min_vol = float(self.min_volume)
                        max_vol = float(self.max_volume)
                        self.smoothed_volume = max(min_vol, min(max_vol, self.smoothed_volume))

                        try:
                            self.set_volume(round(self.smoothed_volume))
                            self.current_volume = self.smoothed_volume
                        except Exception as e:
                            print(f"Volume setting error: {e}")

                    # Auto-exit: check convergence after warmup
                    if self.auto_exit and not self.is_in_warmup:
                        b_stable = (_last_target_brightness is not None and
                                    abs(self.smoothed_brightness - _last_target_brightness) < _converge_threshold)
                        v_stable = (not AUDIO_AVAILABLE or not self.is_audio_in_warmup) and (
                            not AUDIO_AVAILABLE or (
                                _last_target_volume is not None and
                                abs(self.smoothed_volume - _last_target_volume) < _converge_threshold))
                        if b_stable and v_stable:
                            _converge_count += 1
                        else:
                            _converge_count = 0
                        _last_target_brightness = target_brightness if camera_brightness is not None else _last_target_brightness
                        _last_target_volume = target_volume if AUDIO_AVAILABLE else _last_target_volume
                        if _converge_count >= _converge_required:
                            elapsed = time.time() - start_time
                            print(f"\nConverged in {elapsed:.1f}s — "
                                  f"brightness: {self.smoothed_brightness:.1f}%, "
                                  f"volume: {self.smoothed_volume:.1f}%")
                            break

                    if current_time - last_brightness_change_time > 10:
                        update_interval = min(update_interval * 1.2, 2.0)
                    else:
                        update_interval = max(update_interval / 1.2, 0.1)

                    time.sleep(update_interval)
                else:
                    time.sleep(self.inactivity_check_interval)
                    self.on_activity()
                    if self.is_active and self.cap is None:
                        self.setup_camera()

        except KeyboardInterrupt:
            print("Stopping controller...")
        finally:
            try:
                config_dir = os.path.expanduser("~/.config/adaptive-controller")
                os.makedirs(config_dir, exist_ok=True)
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
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == '__main__':
    os.makedirs(os.path.expanduser("~/.cache/adaptive-controller"), exist_ok=True)

    if not AUDIO_AVAILABLE:
        print("\nAudio features are disabled. To enable audio support:")
        if platform.system().lower() == "linux":
            print("  For sounddevice: sudo dnf install portaudio-devel && pip install sounddevice --user")
            print("  Alternative methods: sudo dnf install alsa-utils sox")
        else:
            print("  pip install sounddevice --user")
        print("\nContinuing without audio features...\n")

    if not SCREEN_CAPTURE_AVAILABLE:
        print("\nScreen content analysis is disabled. To enable, install one of:")
        print("  pip install mss --user        # Preferred method")
        print("  pip install pillow --user     # Alternative method")
        print("\nContinuing without screen content analysis...\n")

    continuous = "--continuous" in sys.argv

    try:
        controller = AdaptiveBrightnessVolumeController(auto_exit=not continuous)
        print("\nStarting adaptive brightness and volume controller...")
        print(f"Platform: {platform.system()}")
        if continuous:
            print("Mode: continuous (press Ctrl+C to stop)")
        print(f"Brightness range: {controller.min_brightness}% - {controller.max_brightness}%")
        print(f"Volume range: {controller.min_volume}% - {controller.max_volume}%")
        print(f"Brightness control method: {controller.brightness_method}")
        print(f"Exposure lock: {'enabled' if controller.lock_exposure else 'disabled'}")
        if SCREEN_CAPTURE_AVAILABLE:
            print(f"Screen capture method: {SCREEN_CAPTURE_METHOD}")
        if AUDIO_AVAILABLE:
            print(f"Audio control: Enabled ({AUDIO_METHOD})")
        else:
            print("Audio control: Disabled")
        controller.run()
    except KeyboardInterrupt:
        print("\nStopping controller...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
