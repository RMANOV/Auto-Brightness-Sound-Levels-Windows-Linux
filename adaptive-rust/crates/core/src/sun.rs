//! NOAA Solar Calculator
//!
//! Pure math implementation of sunrise/sunset times.
//! Used for sun-aware seasonal adaptation of smoothing parameters.

use std::f64::consts::PI;

/// Sun window type for adaptive behavior
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SunWindow {
    Sunrise,
    Sunset,
}

fn deg(r: f64) -> f64 { r * 180.0 / PI }
fn rad(d: f64) -> f64 { d * PI / 180.0 }

fn julian_day(year: i32, month: u32, day: u32) -> f64 {
    let a = (14 - month as i32) / 12;
    let y = year + 4800 - a;
    let m = month as i32 + 12 * a - 3;
    day as f64 + (153 * m + 2) as f64 / 5.0
        + 365.0 * y as f64 + (y / 4) as f64
        - (y / 100) as f64 + (y / 400) as f64 - 32045.0
}

fn solar_mean_anomaly(t: f64) -> f64 {
    357.52911 + t * (35999.05029 - 0.0001537 * t)
}

fn sun_eq_center(t: f64) -> f64 {
    let m = rad(solar_mean_anomaly(t));
    m.sin() * (1.914602 - t * (0.004817 + 0.000014 * t))
        + (2.0 * m).sin() * (0.019993 - 0.000101 * t)
        + (3.0 * m).sin() * 0.000289
}

fn sun_true_lon(t: f64) -> f64 {
    280.46646 + t * (36000.76983 + 0.0003032 * t) + sun_eq_center(t)
}

fn sun_apparent_lon(t: f64) -> f64 {
    let omega = 125.04 - 1934.136 * t;
    sun_true_lon(t) - 0.00569 - 0.00478 * rad(omega).sin()
}

fn obliquity_correction(t: f64) -> f64 {
    let e0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let omega = 125.04 - 1934.136 * t;
    e0 + 0.00256 * rad(omega).cos()
}

fn sun_declination(t: f64) -> f64 {
    let e = rad(obliquity_correction(t));
    let lambda = rad(sun_apparent_lon(t));
    deg((e.sin() * lambda.sin()).asin())
}

fn equation_of_time(t: f64) -> f64 {
    let eps = rad(obliquity_correction(t));
    let l0 = rad(sun_true_lon(t));
    let e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
    let m = rad(solar_mean_anomaly(t));
    let y = (eps / 2.0).tan().powi(2);

    let etime = y * (2.0 * l0).sin()
        - 2.0 * e * m.sin()
        + 4.0 * e * y * m.sin() * (2.0 * l0).cos()
        - 0.5 * y * y * (4.0 * l0).sin()
        - 1.25 * e * e * (2.0 * m).sin();

    deg(etime) * 4.0
}

fn hour_angle_sunrise(latitude: f64, solar_dec: f64) -> f64 {
    let ha_arg = rad(90.833).cos() / (rad(latitude).cos() * rad(solar_dec).cos())
        - rad(latitude).tan() * rad(solar_dec).tan();
    if ha_arg < -1.0 {
        180.0
    } else if ha_arg > 1.0 {
        0.0
    } else {
        deg(ha_arg.acos())
    }
}

/// Calculate sunrise/sunset as minutes from midnight (local time).
pub fn calculate_sun_times(
    latitude: f64, longitude: f64, timezone_offset: f64,
    year: i32, month: u32, day: u32,
) -> (f64, f64) {
    let jd = julian_day(year, month, day);
    let t = (jd - 2451545.0) / 36525.0;

    let solar_dec = sun_declination(t);
    let eq_time = equation_of_time(t);
    let ha = hour_angle_sunrise(latitude, solar_dec);

    let correction = eq_time + 4.0 * longitude;
    let noon_utc = 720.0 - correction;
    let off = timezone_offset * 60.0;

    let sunrise = ((noon_utc - 4.0 * ha + off) % 1440.0 + 1440.0) % 1440.0;
    let sunset = ((noon_utc + 4.0 * ha + off) % 1440.0 + 1440.0) % 1440.0;
    (sunrise, sunset)
}

/// Detect if current time falls in a sunrise/sunset window.
/// Window: 30 min before to 2 h after each event.
pub fn detect_sun_window(
    current_minutes: f64,
    latitude: f64, longitude: f64, timezone_offset: f64,
    year: i32, month: u32, day: u32,
) -> Option<SunWindow> {
    let (sunrise, sunset) = calculate_sun_times(
        latitude, longitude, timezone_offset, year, month, day,
    );
    if current_minutes >= sunrise - 30.0 && current_minutes <= sunrise + 120.0 {
        Some(SunWindow::Sunrise)
    } else if current_minutes >= sunset - 30.0 && current_minutes <= sunset + 120.0 {
        Some(SunWindow::Sunset)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sofia_feb() {
        // Sofia, Feb 8 — sunrise ~07:30, sunset ~17:48
        let (sr, ss) = calculate_sun_times(42.6977, 23.3219, 2.0, 2026, 2, 8);
        let sr_h = sr / 60.0;
        let ss_h = ss / 60.0;
        assert!(sr_h > 7.0 && sr_h < 8.0, "sunrise {sr_h:.2}h");
        assert!(ss_h > 17.0 && ss_h < 18.5, "sunset {ss_h:.2}h");
    }

    #[test]
    fn test_window_detection() {
        // 07:40 (460 min) should be in sunrise window for Sofia Feb
        let w = detect_sun_window(460.0, 42.6977, 23.3219, 2.0, 2026, 2, 8);
        assert_eq!(w, Some(SunWindow::Sunrise));

        // 12:00 (720 min) should be outside any window
        let w = detect_sun_window(720.0, 42.6977, 23.3219, 2.0, 2026, 2, 8);
        assert_eq!(w, None);
    }
}
