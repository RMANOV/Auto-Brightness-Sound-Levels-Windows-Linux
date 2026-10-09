//! Ambient light source
//!
//! Windows: uses NOAA sun-position algorithm to simulate ambient light curve.
//! Provides smooth brightness transitions at sunrise/sunset based on solar angle.
//! Hardcoded for Sofia, Bulgaria (42.6977°N, 23.3219°E, UTC+2).

use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

/// Sofia, Bulgaria coordinates
const LATITUDE: f64 = 42.6977;
const LONGITUDE: f64 = 23.3219;
const TIMEZONE_OFFSET: f64 = 2.0; // UTC+2 (EET); TODO: detect DST for EEST (UTC+3)

/// Ambient light source based on sun position
pub struct Camera {
    latitude: f64,
    longitude: f64,
    timezone_offset: f64,
}

impl Camera {
    pub fn new(_index: usize) -> Result<Self> {
        info!(
            "Sun-position ambient light (Sofia: {:.4}°N, {:.4}°E, UTC+{:.0})",
            LATITUDE, LONGITUDE, TIMEZONE_OFFSET
        );
        Ok(Self {
            latitude: LATITUDE,
            longitude: LONGITUDE,
            timezone_offset: TIMEZONE_OFFSET,
        })
    }

    /// Return a simulated grayscale frame whose pixel value reflects current sun position
    pub fn capture_frame(&mut self) -> Result<Vec<u8>> {
        let unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let local = unix + (self.timezone_offset * 3600.0) as i64;
        let current_min = local.rem_euclid(86400) as f64 / 60.0;
        let (year, month, day) = civil_from_days(local.div_euclid(86400));

        let (sunrise, sunset) = adaptive_core::sun::calculate_sun_times(
            self.latitude,
            self.longitude,
            self.timezone_offset,
            year,
            month,
            day,
        );

        let brightness_pct = sun_brightness_curve(current_min, sunrise, sunset);
        let pixel_val = (brightness_pct * 2.55).clamp(0.0, 255.0) as u8;

        // Simulated 320x240 grayscale frame
        Ok(vec![pixel_val; 320 * 240])
    }
}

/// Smooth S-curve brightness based on time relative to sunrise/sunset.
///
/// Returns 0-100 percentage:
/// - Night (> 1h after sunset, > 1h before sunrise): 5%
/// - Dawn/dusk transition: smooth cosine interpolation
/// - Full daylight: 80%
fn sun_brightness_curve(current_min: f64, sunrise: f64, sunset: f64) -> f64 {
    let dawn_start = sunrise - 60.0;
    let dawn_end = sunrise + 60.0;
    let dusk_start = sunset - 60.0;
    let dusk_end = sunset + 60.0;

    if current_min < dawn_start || current_min > dusk_end {
        // Night
        5.0
    } else if current_min >= dawn_end && current_min <= dusk_start {
        // Full daylight
        80.0
    } else if current_min >= dawn_start && current_min <= dawn_end {
        // Dawn transition (cosine interpolation)
        let progress = (current_min - dawn_start) / (dawn_end - dawn_start);
        let smooth = 0.5 * (1.0 - (std::f64::consts::PI * progress).cos());
        5.0 + smooth * 75.0
    } else {
        // Dusk transition
        let progress = (current_min - dusk_start) / (dusk_end - dusk_start);
        let smooth = 0.5 * (1.0 - (std::f64::consts::PI * progress).cos());
        80.0 - smooth * 75.0
    }
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
