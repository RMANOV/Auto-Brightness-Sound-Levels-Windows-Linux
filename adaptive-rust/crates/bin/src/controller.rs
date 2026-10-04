//! Main controller implementation
//!
//! Coordinates camera/sun capture, audio analysis, and system control.

use adaptive_core::sun::SunWindow;
use adaptive_core::{
    calculate_brightness, calculate_brightness_mapping, calculate_volume_mapping,
    check_significant_change, compute_noise_level, smooth_transition,
};
use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::audio_capture::AudioCapture;
use crate::camera::Camera;
use crate::system::{BrightnessControl, VolumeControl};

/// Sofia, Bulgaria
const LATITUDE: f64 = 42.6977;
const LONGITUDE: f64 = 23.3219;
const TIMEZONE_OFFSET: f64 = 2.0;

/// Controller configuration
#[derive(Clone)]
pub struct ControllerConfig {
    pub min_brightness: f32,
    pub max_brightness: f32,
    pub min_volume: f32,
    pub max_volume: f32,
    pub brightness_smoothing: f32,
    pub volume_smoothing: f32,
    pub update_interval: Duration,
    pub warmup_frames: u32,
    pub auto_exit: bool,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            min_brightness: 5.0,
            max_brightness: 45.0,
            min_volume: 3.0,
            max_volume: 35.0,
            brightness_smoothing: 0.3,
            volume_smoothing: 0.2,
            update_interval: Duration::from_millis(500),
            warmup_frames: 20,
            auto_exit: true,
        }
    }
}

/// Frame data from camera / sun-position
pub struct FrameData {
    pub brightness: f32,
}

/// Audio data from capture
pub struct AudioData {
    pub noise_level: f32,
}

/// Main controller
pub struct Controller {
    config: ControllerConfig,

    // State
    current_brightness: f32,
    smoothed_brightness: f32,
    current_volume: f32,
    smoothed_volume: f32,
    last_camera_brightness: Option<f32>,
    warmup_frame: u32,
    last_significant_change: Instant,

    // Channels for async data
    brightness_rx: Receiver<FrameData>,
    audio_rx: Receiver<AudioData>,

    // Thread handles
    camera_thread: Option<thread::JoinHandle<()>>,
    audio_thread: Option<thread::JoinHandle<()>>,

    // Shutdown flag
    shutdown: Arc<Mutex<bool>>,

    // System controls
    brightness_control: BrightnessControl,
    volume_control: VolumeControl,

    // Timing
    last_update: Instant,
    last_perf_report: Instant,
    start_time: Instant,

    // Auto-exit convergence tracking
    last_target_brightness: Option<f32>,
    last_target_volume: Option<f32>,
    converge_count: u32,

    // Sun-aware seasonal adaptation
    sun_window: Option<SunWindow>,
}

impl Controller {
    pub fn new(config: ControllerConfig) -> Result<Self> {
        let shutdown = Arc::new(Mutex::new(false));

        // Create channels
        let (brightness_tx, brightness_rx) = bounded(10);
        let (audio_tx, audio_rx) = bounded(10);

        // Initialize system controls
        let brightness_control = BrightnessControl::new()?;
        let volume_control = VolumeControl::new()?;

        // Get initial values
        let current_brightness = brightness_control.get()? as f32;
        let current_volume = volume_control.get()? as f32;

        info!("Initial brightness: {}%", current_brightness);
        info!("Initial volume: {}%", current_volume);

        // Detect sun window for seasonal adaptation (NOAA algorithm)
        let sun_window = detect_sun_window_now();
        if let Some(ref w) = sun_window {
            info!("Sun-aware mode: {:?} window — accelerated adaptation", w);
        } else {
            info!("Outside sunrise/sunset window — normal mode");
        }

        // Spawn camera thread (sun-position based on Windows)
        let shutdown_clone = Arc::clone(&shutdown);
        let camera_thread = Some(thread::spawn(move || {
            camera_worker(brightness_tx, shutdown_clone);
        }));

        // Spawn audio thread
        let shutdown_clone = Arc::clone(&shutdown);
        let audio_thread = Some(thread::spawn(move || {
            audio_worker(audio_tx, shutdown_clone);
        }));

        Ok(Self {
            config,
            current_brightness,
            smoothed_brightness: current_brightness,
            current_volume,
            smoothed_volume: current_volume,
            last_camera_brightness: None,
            warmup_frame: 0,
            last_significant_change: Instant::now(),
            brightness_rx,
            audio_rx,
            camera_thread,
            audio_thread,
            shutdown,
            brightness_control,
            volume_control,
            last_update: Instant::now(),
            last_perf_report: Instant::now(),
            start_time: Instant::now(),
            last_target_brightness: None,
            last_target_volume: None,
            converge_count: 0,
            sun_window,
        })
    }

    /// Process one tick. Returns true if converged (auto-exit).
    pub fn tick(&mut self) -> Result<bool> {
        let now = Instant::now();

        if now.duration_since(self.last_update) < self.config.update_interval {
            thread::sleep(Duration::from_millis(10));
            return Ok(false);
        }
        self.last_update = now;

        let prev_target_b = self.last_target_brightness;
        let prev_target_v = self.last_target_volume;

        self.process_brightness()?;
        self.process_audio()?;

        // Auto-exit convergence check (after warmup)
        if self.config.auto_exit && self.warmup_frame >= self.config.warmup_frames {
            let b_ok = prev_target_b.map_or(false, |t| (self.smoothed_brightness - t).abs() < 1.0);
            let v_ok = prev_target_v.map_or(true, |t| (self.smoothed_volume - t).abs() < 1.0);
            if b_ok && v_ok {
                self.converge_count += 1;
            } else {
                self.converge_count = 0;
            }
            if self.converge_count >= 3 {
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let window_info = self
                    .sun_window
                    .map_or(String::new(), |w| format!(" ({:?} window)", w));
                info!(
                    "Converged in {:.1}s{} — brightness: {:.1}%, volume: {:.1}%",
                    elapsed, window_info, self.smoothed_brightness, self.smoothed_volume
                );
                return Ok(true);
            }
        }

        // Periodic performance report
        if now.duration_since(self.last_perf_report) > Duration::from_secs(30) {
            self.print_performance_stats();
            self.last_perf_report = now;
        }

        Ok(false)
    }

    fn process_brightness(&mut self) -> Result<()> {
        let mut latest: Option<FrameData> = None;
        loop {
            match self.brightness_rx.try_recv() {
                Ok(data) => latest = Some(data),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    warn!("Camera thread disconnected");
                    break;
                }
            }
        }

        if let Some(frame_data) = latest {
            let camera_brightness = frame_data.brightness;

            if let Some(last) = self.last_camera_brightness {
                let is_dimming = camera_brightness < last;
                if check_significant_change(camera_brightness, last, is_dimming) {
                    self.last_significant_change = Instant::now();
                    let direction = if is_dimming { "DIMMING" } else { "BRIGHTENING" };
                    info!(
                        "Light change: {} {:.1} -> {:.1}",
                        direction, last, camera_brightness
                    );
                }
            }
            self.last_camera_brightness = Some(camera_brightness);

            let target = calculate_brightness_mapping(
                camera_brightness,
                self.config.min_brightness,
                self.config.max_brightness,
            );

            self.last_target_brightness = Some(target);

            let time_since_change = self.last_significant_change.elapsed().as_secs_f32();
            let sun_boost = if self.sun_window.is_some() {
                1.5
            } else {
                1.0
            };
            let smooth_factor = if self.warmup_frame < self.config.warmup_frames {
                self.warmup_frame += 1;
                self.config.brightness_smoothing * 0.05
            } else if time_since_change < 5.0 {
                self.config.brightness_smoothing * 2.0 * sun_boost
            } else {
                self.config.brightness_smoothing * sun_boost
            };

            self.smoothed_brightness = smooth_transition(
                self.smoothed_brightness,
                target,
                smooth_factor,
            )
            .clamp(self.config.min_brightness, self.config.max_brightness);

            let new_brightness = self.smoothed_brightness.round();
            if (new_brightness - self.current_brightness).abs() >= 1.0 {
                debug!(
                    "Setting brightness: {:.1}% -> {:.1}%",
                    self.current_brightness, new_brightness
                );
                self.brightness_control.set(new_brightness as i32)?;
                self.current_brightness = new_brightness;
            }
        }

        Ok(())
    }

    fn process_audio(&mut self) -> Result<()> {
        let mut latest: Option<AudioData> = None;
        loop {
            match self.audio_rx.try_recv() {
                Ok(data) => latest = Some(data),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        if let Some(audio_data) = latest {
            const MIN_NOISE: f32 = 5e-6;
            const MAX_NOISE: f32 = 8e-3;
            let normalized = ((audio_data.noise_level - MIN_NOISE) / (MAX_NOISE - MIN_NOISE))
                .clamp(0.0, 1.0);

            let target = calculate_volume_mapping(
                normalized,
                self.config.min_volume,
                self.config.max_volume,
            );

            self.last_target_volume = Some(target);

            let vol_smooth = self.config.volume_smoothing
                * if self.sun_window.is_some() { 1.5 } else { 1.0 };
            self.smoothed_volume = smooth_transition(self.smoothed_volume, target, vol_smooth)
                .clamp(self.config.min_volume, self.config.max_volume);

            let new_volume = self.smoothed_volume.round();
            if (new_volume - self.current_volume).abs() >= 1.0 {
                debug!(
                    "Setting volume: {:.1}% -> {:.1}%",
                    self.current_volume, new_volume
                );
                self.volume_control.set(new_volume as i32)?;
                self.current_volume = new_volume;
            }
        }

        Ok(())
    }

    fn print_performance_stats(&self) {
        info!(
            "Status: brightness={:.1}%, volume={:.1}%, uptime={:.0}s",
            self.current_brightness,
            self.current_volume,
            self.start_time.elapsed().as_secs_f32()
        );
    }

    pub fn cleanup(&mut self) {
        info!("Cleaning up controller...");
        *self.shutdown.lock() = true;

        if let Some(handle) = self.camera_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.audio_thread.take() {
            let _ = handle.join();
        }

        info!("Controller cleanup complete");
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Camera worker thread
fn camera_worker(tx: Sender<FrameData>, shutdown: Arc<Mutex<bool>>) {
    let mut camera = match Camera::new(0) {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to initialize ambient light source: {}", e);
            return;
        }
    };

    info!("Camera worker started");

    while !*shutdown.lock() {
        match camera.capture_frame() {
            Ok(frame) => {
                let brightness = calculate_brightness(&frame);
                let data = FrameData { brightness };
                if tx.send(data).is_err() {
                    break;
                }
            }
            Err(e) => {
                debug!("Frame capture error: {}", e);
            }
        }
        // Sun position changes slowly — 2s between samples is plenty
        thread::sleep(Duration::from_secs(2));
    }

    info!("Camera worker stopped");
}

/// Audio worker thread
///
/// Samples ambient noise for 200ms every 5 seconds.
/// Microphone is OFF ~96% of the time (privacy + battery).
fn audio_worker(tx: Sender<AudioData>, shutdown: Arc<Mutex<bool>>) {
    let capture = match AudioCapture::new() {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to initialize audio: {}", e);
            return;
        }
    };

    info!("Audio worker started (sampling 200ms every 5s)");

    while !*shutdown.lock() {
        match capture.capture_samples(Duration::from_millis(200)) {
            Ok(samples) => {
                let noise_level = compute_noise_level(&samples);
                let data = AudioData { noise_level };
                if tx.send(data).is_err() {
                    break;
                }
            }
            Err(e) => {
                debug!("Audio capture error: {}", e);
            }
        }
        // Microphone OFF for 5 seconds between samples
        thread::sleep(Duration::from_secs(5));
    }

    info!("Audio worker stopped");
}

/// Detect current sun window using hardcoded Sofia coordinates + NOAA algorithm
fn detect_sun_window_now() -> Option<SunWindow> {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    let local = unix + (TIMEZONE_OFFSET * 3600.0) as i64;
    let current_min = local.rem_euclid(86400) as f64 / 60.0;
    let (year, month, day) = civil_from_days(local.div_euclid(86400));

    let (sunrise, sunset) = adaptive_core::sun::calculate_sun_times(
        LATITUDE,
        LONGITUDE,
        TIMEZONE_OFFSET,
        year,
        month,
        day,
    );

    info!(
        "Today: sunrise {:02}:{:02}, sunset {:02}:{:02} (Sofia)",
        (sunrise / 60.0) as u32,
        (sunrise % 60.0) as u32,
        (sunset / 60.0) as u32,
        (sunset % 60.0) as u32
    );
    info!(
        "Current time: {:02}:{:02}",
        (current_min / 60.0) as u32,
        (current_min % 60.0) as u32
    );

    adaptive_core::sun::detect_sun_window(
        current_min, LATITUDE, LONGITUDE, TIMEZONE_OFFSET, year, month, day,
    )
}

/// Howard Hinnant's algorithm: days since Unix epoch → (year, month, day)
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}
