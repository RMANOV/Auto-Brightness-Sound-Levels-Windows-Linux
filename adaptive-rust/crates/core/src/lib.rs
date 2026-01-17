//! Adaptive Brightness/Volume Controller Core Library
//!
//! High-performance SIMD-optimized computations for ambient light and audio analysis.
//! Target: 2-4x faster than Python+Numba JIT baseline.

pub mod audio;
pub mod brightness;
pub mod change;
pub mod screen;
pub mod smooth;
pub mod volume;

pub use audio::compute_noise_level;
pub use brightness::{calculate_brightness, calculate_brightness_mapping};
pub use change::check_significant_change;
pub use screen::analyze_screen_brightness;
pub use smooth::smooth_transition;
pub use volume::calculate_volume_mapping;

/// Configuration constants
pub mod config {
    /// Default minimum brightness percentage
    pub const MIN_BRIGHTNESS: f32 = 5.0;
    /// Default maximum brightness percentage
    pub const MAX_BRIGHTNESS: f32 = 45.0;
    /// Default minimum volume percentage
    pub const MIN_VOLUME: f32 = 3.0;
    /// Default maximum volume percentage
    pub const MAX_VOLUME: f32 = 35.0;
    /// Brightness change threshold for dimming
    pub const DIMMING_THRESHOLD: f32 = 8.0;
    /// Brightness change threshold for brightening
    pub const BRIGHTENING_THRESHOLD: f32 = 12.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_noise_level() {
        let audio: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2, 0.0];
        let rms = compute_noise_level(&audio);
        assert!((rms - 0.1414).abs() < 0.01);
    }

    #[test]
    fn test_brightness_mapping_dark() {
        // Dark room (0% camera) should map to min brightness
        let result = calculate_brightness_mapping(0.0, 5.0, 45.0);
        assert!((result - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_brightness_mapping_bright() {
        // Bright room (100% camera) should map to max brightness
        let result = calculate_brightness_mapping(100.0, 5.0, 45.0);
        assert!((result - 40.0).abs() < 1.0);
    }

    #[test]
    fn test_brightness_mapping_boost() {
        // Middle values (45%) should get boost
        let result = calculate_brightness_mapping(45.0, 5.0, 45.0);
        // Base would be 5 + 45*0.35 = 20.75, with 1.35x boost = 28.0
        assert!(result > 25.0 && result < 32.0);
    }

    #[test]
    fn test_volume_mapping() {
        let result = calculate_volume_mapping(0.5, 3.0, 35.0);
        assert!(result > 3.0 && result < 35.0);
    }

    #[test]
    fn test_smooth_transition() {
        let result = smooth_transition(10.0, 20.0, 0.3);
        assert!((result - 13.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_screen() {
        // Bright screen (200 avg) should return > 1.0
        let bright: Vec<u8> = vec![200; 100];
        let factor = analyze_screen_brightness(&bright);
        assert!(factor > 1.0);

        // Dark screen (50 avg) should return < 1.0
        let dark: Vec<u8> = vec![50; 100];
        let factor = analyze_screen_brightness(&dark);
        assert!(factor < 1.0);
    }

    #[test]
    fn test_significant_change() {
        // Large dimming change should be detected
        assert!(check_significant_change(30.0, 50.0, true));
        // Small change should not be detected
        assert!(!check_significant_change(48.0, 50.0, true));
    }
}
