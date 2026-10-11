//! Main controller implementation
//!
//! Coordinates camera/sun capture, audio analysis, and system control.

use adaptive_core::sun::SunWindow;
use adaptive_core::{
    calculate_brightness, calculate_brightness_mapping, calculate_volume_mapping,
    check_significant_change, compute_noise_level, smooth_transition,
};
use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError, TrySendError};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::audio_capture::AudioCapture;
use crate::camera::Camera;
use crate::system::{BrightnessControl, BrightnessReadback, VolumeControl};

/// Sofia, Bulgaria
const LATITUDE: f64 = 42.6977;
const LONGITUDE: f64 = 23.3219;
#[cfg(target_os = "windows")]
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
    convergence: Convergence,

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
        // Unavailable capability has no confirmed percent; zero only seeds unused smoothing.
        let current_brightness = brightness_control
            .get_readback()?
            .map_or(0.0, |r| r.percent as f32);
        let current_volume = volume_control.get()? as f32;

        if brightness_control.is_available() {
            info!("Initial brightness: {}%", current_brightness);
        } else {
            info!("Brightness unavailable — volume-only adaptation");
        }
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
        let camera_thread = if brightness_control.is_available() {
            Some(thread::spawn(move || {
                camera_worker(brightness_tx, shutdown_clone)
            }))
        } else {
            drop(brightness_tx);
            None
        };

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
            convergence: Convergence::default(),
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

        let fresh_brightness = self.process_brightness()?;
        let fresh_audio = self.process_audio()?;
        let brightness_available = self.brightness_control.is_available();
        if !brightness_available && fresh_audio && self.warmup_frame < self.config.warmup_frames {
            self.warmup_frame += 1;
        }

        // Auto-exit convergence check (after warmup)
        if self.config.auto_exit
            && self.warmup_frame >= self.config.warmup_frames
            && self
                .convergence
                .step_ready(brightness_available, fresh_brightness, fresh_audio)
        {
            let b_ok = if let Some(reading) = self.brightness_control.get_readback()? {
                self.current_brightness = reading.percent as f32;
                confirmed_brightness_stable(reading, self.last_target_brightness)
            } else {
                true // explicitly unavailable, never a measured/applied brightness
            };
            self.current_volume = self.volume_control.get()? as f32;
            let v_ok = confirmed_volume_stable(self.current_volume, self.last_target_volume);
            if self.convergence.record_step(b_ok && v_ok) {
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let window_info = self
                    .sun_window
                    .map_or(String::new(), |w| format!(" ({:?} window)", w));
                if brightness_available {
                    info!(
                        "Converged in {:.1}s{} — brightness: {:.1}%, volume: {:.1}%",
                        elapsed, window_info, self.current_brightness, self.current_volume
                    );
                } else {
                    info!(
                        "Converged in {:.1}s{} — brightness: unavailable, volume only: {:.1}%",
                        elapsed, window_info, self.current_volume
                    );
                }
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

    fn process_brightness(&mut self) -> Result<bool> {
        if !self.brightness_control.is_available() {
            return Ok(false);
        }
        let latest = drain_latest(&self.brightness_rx, "camera")?;
        let fresh = latest.is_some();
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
            let sun_boost = if self.sun_window.is_some() { 1.5 } else { 1.0 };
            let smooth_factor = if self.warmup_frame < self.config.warmup_frames {
                self.warmup_frame += 1;
                self.config.brightness_smoothing * 0.05
            } else if time_since_change < 5.0 {
                self.config.brightness_smoothing * 2.0 * sun_boost
            } else {
                self.config.brightness_smoothing * sun_boost
            };

            self.smoothed_brightness =
                smooth_transition(self.smoothed_brightness, target, smooth_factor)
                    .clamp(self.config.min_brightness, self.config.max_brightness);

            let new_brightness = self.smoothed_brightness.round();
            if (new_brightness - self.current_brightness).abs() >= 1.0 {
                debug!(
                    "Setting brightness: {:.1}% -> {:.1}%",
                    self.current_brightness, new_brightness
                );
                self.brightness_control.set(new_brightness as i32)?;
                self.current_brightness = self.brightness_control.get()? as f32;
            }
        }

        Ok(fresh)
    }

    fn process_audio(&mut self) -> Result<bool> {
        let latest = latest_audio(&self.audio_rx)?;

        let fresh = latest.is_some();
        if let Some(audio_data) = latest {
            const MIN_NOISE: f32 = 5e-6;
            const MAX_NOISE: f32 = 8e-3;
            let normalized =
                ((audio_data.noise_level - MIN_NOISE) / (MAX_NOISE - MIN_NOISE)).clamp(0.0, 1.0);

            let target = calculate_volume_mapping(
                normalized,
                self.config.min_volume,
                self.config.max_volume,
            );

            self.last_target_volume = Some(target);

            let vol_smooth =
                self.config.volume_smoothing * if self.sun_window.is_some() { 1.5 } else { 1.0 };
            self.smoothed_volume = smooth_transition(self.smoothed_volume, target, vol_smooth)
                .clamp(self.config.min_volume, self.config.max_volume);

            let new_volume = self.smoothed_volume.round();
            if (new_volume - self.current_volume).abs() >= 1.0 {
                debug!(
                    "Setting volume: {:.1}% -> {:.1}%",
                    self.current_volume, new_volume
                );
                self.volume_control.set(new_volume as i32)?;
                self.current_volume = self.volume_control.get()? as f32;
            }
        }

        Ok(fresh)
    }

    fn print_performance_stats(&self) {
        if self.brightness_control.is_available() {
            info!(
                "Status: brightness={:.1}%, volume={:.1}%, uptime={:.0}s",
                self.current_brightness,
                self.current_volume,
                self.start_time.elapsed().as_secs_f32()
            );
        } else {
            info!(
                "Status: brightness=unavailable, volume={:.1}%, uptime={:.0}s",
                self.current_volume,
                self.start_time.elapsed().as_secs_f32()
            );
        }
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
                match tx.try_send(data) {
                    Ok(()) | Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
            Err(e) => {
                debug!("Frame capture error: {}", e);
            }
        }
        // Restore Linux camera cadence; preserve the Windows solar simulator cadence.
        #[cfg(target_os = "linux")]
        thread::sleep(Duration::from_millis(100));
        #[cfg(target_os = "windows")]
        pause_unless_shutdown(&shutdown, Duration::from_secs(2));
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
                match tx.try_send(data) {
                    Ok(()) | Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
            Err(e) => {
                debug!("Audio capture error: {}", e);
            }
        }
        // Microphone OFF for 5 seconds between samples
        pause_unless_shutdown(&shutdown, Duration::from_secs(5));
    }

    info!("Audio worker stopped");
}

/// Sleeps for `duration`, waking within 100ms of a shutdown request so cleanup
/// never waits out a whole capture interval. Convergence is decided right after
/// an audio sample, i.e. at the start of the audio worker's longest pause.
fn pause_unless_shutdown(shutdown: &Mutex<bool>, duration: Duration) {
    let deadline = Instant::now() + duration;
    while !*shutdown.lock() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        thread::sleep(remaining.min(Duration::from_millis(100)));
    }
}

/// Detect current sun window using hardcoded Sofia coordinates + NOAA algorithm
fn detect_sun_window_now() -> Option<SunWindow> {
    let unix = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    #[cfg(target_os = "windows")]
    let timezone_offset = TIMEZONE_OFFSET;
    #[cfg(target_os = "linux")]
    let timezone_offset = match linux_timezone_offset(unix) {
        Ok(offset) => offset,
        Err(error) => {
            warn!("Sun boost disabled: {error}");
            return None;
        }
    };
    let local = unix + (timezone_offset * 3600.0) as i64;
    let current_min = local.rem_euclid(86400) as f64 / 60.0;
    let (year, month, day) = civil_from_days(local.div_euclid(86400));

    let (sunrise, sunset) = adaptive_core::sun::calculate_sun_times(
        LATITUDE,
        LONGITUDE,
        timezone_offset,
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
        current_min,
        LATITUDE,
        LONGITUDE,
        timezone_offset,
        year,
        month,
        day,
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

/// Takes the newest queued sample. A disconnected required channel is an error
/// once its queue is empty, while an ordinarily empty live channel is a valid poll.
fn drain_latest<T>(rx: &Receiver<T>, channel: &str) -> Result<Option<T>> {
    let mut latest = None;
    loop {
        match rx.try_recv() {
            Ok(data) => latest = Some(data),
            Err(TryRecvError::Empty) => return Ok(latest),
            Err(TryRecvError::Disconnected) if latest.is_some() => return Ok(latest),
            Err(TryRecvError::Disconnected) => {
                anyhow::bail!("Required {channel} channel disconnected")
            }
        }
    }
}

fn latest_audio(rx: &Receiver<AudioData>) -> Result<Option<AudioData>> {
    let latest = drain_latest(rx, "audio")?;
    if let Some(data) = &latest {
        anyhow::ensure!(
            data.noise_level.is_finite() && data.noise_level >= 0.0,
            "Invalid audio measurement"
        );
    }
    Ok(latest)
}

/// Auto-exit counts measured stability steps, not poll ticks. Camera and audio
/// arrive on independent cadences, so fresh evidence accumulates across polls and
/// each step consumes it once: no stream can advance the count on a stale sample.
#[derive(Default)]
struct Convergence {
    fresh_camera: bool,
    fresh_audio: bool,
    stable_steps: u32,
}

impl Convergence {
    /// Records this poll's samples; true when every required stream has new evidence.
    fn step_ready(
        &mut self,
        brightness_available: bool,
        fresh_camera: bool,
        fresh_audio: bool,
    ) -> bool {
        self.fresh_camera |= fresh_camera;
        self.fresh_audio |= fresh_audio;
        let ready = self.fresh_audio && (self.fresh_camera || !brightness_available);
        if ready {
            self.fresh_camera = false;
            self.fresh_audio = false;
        }
        ready
    }

    /// Scores a ready step; any mismatch restarts the count. True once converged.
    fn record_step(&mut self, stable: bool) -> bool {
        self.stable_steps = if stable { self.stable_steps + 1 } else { 0 };
        self.stable_steps >= 3
    }
}

fn confirmed_brightness_stable(actual: BrightnessReadback, target: Option<f32>) -> bool {
    target.is_some_and(|t| actual.matches_target(t))
}

fn confirmed_volume_stable(actual: f32, target: Option<f32>) -> bool {
    actual.is_finite()
        && (0.0..=100.0).contains(&actual)
        && target
            .is_some_and(|t| t.is_finite() && (actual - t.round().clamp(0.0, 100.0)).abs() <= 1.0)
}

#[cfg(target_os = "linux")]
fn linux_timezone_offset(unix: i64) -> Result<f64> {
    let stamp = format!("@{unix}");
    let output = crate::system::command_linux::output("/usr/bin/date", &["--date", &stamp, "+%z"])?;
    parse_timezone_offset(output.trim())
}

#[cfg(target_os = "linux")]
fn parse_timezone_offset(text: &str) -> Result<f64> {
    anyhow::ensure!(
        text.len() == 5
            && (text.starts_with('+') || text.starts_with('-'))
            && text[1..].bytes().all(|x| x.is_ascii_digit()),
        "Invalid local UTC offset"
    );
    let hours: i32 = text[1..3].parse()?;
    let minutes: i32 = text[3..5].parse()?;
    anyhow::ensure!(hours <= 23 && minutes < 60, "Invalid local UTC offset");
    let sign = if text.starts_with('-') { -1.0 } else { 1.0 };
    Ok(sign * (hours as f64 + minutes as f64 / 60.0))
}

#[cfg(test)]
mod restoration_tests {
    use super::*;
    fn reading(percent: f32) -> BrightnessReadback {
        BrightnessReadback {
            percent: percent as f64,
            step_percent: 1.0,
        }
    }
    #[test]
    fn supported_brightness_needs_fresh_camera_and_audio_each_step() {
        let mut convergence = Convergence::default();
        // Disjoint polls, including the Windows 2s camera / 5s audio phases.
        let ready: Vec<bool> = [
            (true, false),
            (false, true),
            (true, false),
            (false, false),
            (false, true),
            (false, true),
            (true, false),
        ]
        .into_iter()
        .map(|(camera, audio)| convergence.step_ready(true, camera, audio))
        .collect();
        assert_eq!(ready, [false, true, false, false, true, false, true]);
        // A camera stream alone never makes a step; the next audio sample does.
        for _ in 0..20 {
            assert!(!convergence.step_ready(true, true, false));
        }
        assert!(convergence.step_ready(true, false, true));
        assert!(!convergence.step_ready(true, false, true));
    }
    #[test]
    fn unavailable_brightness_steps_on_fresh_audio_alone() {
        let mut convergence = Convergence::default();
        let ready: Vec<bool> = [true, false, false, true, false, true]
            .into_iter()
            .map(|audio| convergence.step_ready(false, true, audio))
            .collect();
        assert_eq!(ready, [true, false, false, true, false, true]);
    }
    #[test]
    fn convergence_needs_three_consecutive_stable_steps() {
        let mut convergence = Convergence::default();
        let converged: Vec<bool> = [true, true, false, true, true, true]
            .into_iter()
            .map(|stable| convergence.record_step(stable))
            .collect();
        assert_eq!(converged, [false, false, false, false, false, true]);
    }
    #[test]
    fn desired_stability_without_measured_application_is_not_convergence() {
        assert!(confirmed_brightness_stable(reading(30.0), Some(30.0)));
        assert!(!confirmed_brightness_stable(reading(80.0), Some(30.0)));
        assert!(!confirmed_brightness_stable(reading(f32::NAN), Some(30.0)));
        assert!(!confirmed_brightness_stable(
            reading(30.0),
            Some(f32::INFINITY)
        ));
        assert!(!confirmed_brightness_stable(reading(30.0), None));
    }
    #[test]
    fn volume_stability_needs_finite_readback_matching_a_present_target() {
        assert!(confirmed_volume_stable(25.0, Some(25.4)));
        assert!(!confirmed_volume_stable(25.0, None));
        assert!(!confirmed_volume_stable(50.0, Some(25.0)));
        assert!(!confirmed_volume_stable(f32::NAN, Some(25.0)));
        assert!(!confirmed_volume_stable(25.0, Some(f32::NAN)));
    }
    #[test]
    fn coarse_backlight_reaches_convergence_but_a_missed_write_does_not() {
        let coarse = BrightnessReadback {
            percent: 100.0 / 3.0,
            step_percent: 100.0 / 9.0,
        };
        assert!(confirmed_brightness_stable(coarse, Some(30.4)));
        let mismatch = BrightnessReadback {
            percent: 500.0 / 9.0,
            step_percent: 100.0 / 9.0,
        };
        assert!(!confirmed_brightness_stable(mismatch, Some(30.4)));
    }
    #[test]
    fn required_channel_disconnect_is_an_error_after_queued_samples() {
        let (tx, rx) = bounded(2);
        assert!(drain_latest(&rx, "camera").unwrap().is_none());
        tx.send(FrameData { brightness: 10.0 }).unwrap();
        tx.send(FrameData { brightness: 20.0 }).unwrap();
        drop(tx);
        assert_eq!(
            drain_latest(&rx, "camera").unwrap().unwrap().brightness,
            20.0
        );
        assert!(drain_latest(&rx, "camera").is_err());
    }
    #[test]
    fn invalid_audio_measurement_is_not_a_target() {
        for (noise_level, valid) in [
            (0.001, true),
            (f32::NAN, false),
            (f32::INFINITY, false),
            (-0.001, false),
        ] {
            let (tx, rx) = bounded(1);
            tx.send(AudioData { noise_level }).unwrap();
            assert_eq!(latest_audio(&rx).is_ok(), valid, "{noise_level}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn local_zone_offset_preserves_dst_and_fractional_zones() {
        assert_eq!(parse_timezone_offset("+0300").unwrap(), 3.0);
        assert_eq!(parse_timezone_offset("+0200").unwrap(), 2.0);
        assert_eq!(parse_timezone_offset("-0330").unwrap(), -3.5);
        for invalid in ["", "UTC+3", "+2360", "+2400", "nan", "+0é0"] {
            assert!(parse_timezone_offset(invalid).is_err());
        }
    }
}
