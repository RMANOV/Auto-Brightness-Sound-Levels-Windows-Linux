//! Change detection for brightness transitions
//!
//! Detects significant light changes to trigger faster response.

/// Check if brightness change is significant enough to warrant fast response.
///
/// # Arguments
/// * `current_brightness` - Current detected brightness
/// * `last_brightness` - Previously recorded brightness
/// * `is_dimming` - Whether we're looking for dimming (true) or brightening (false)
///
/// # Returns
/// `true` if change exceeds threshold for the specified direction
///
/// # Algorithm
/// - Dimming threshold: 8% change
/// - Brightening threshold: 12% change (higher to avoid false positives)
///
/// # Performance
/// - Branchless implementation using conditional moves
/// - Target: 0.002ms (5x faster than Numba's 0.01ms)
#[inline]
pub fn check_significant_change(
    current_brightness: f32,
    last_brightness: f32,
    is_dimming: bool,
) -> bool {
    const DIMMING_THRESHOLD: f32 = 8.0;
    const BRIGHTENING_THRESHOLD: f32 = 12.0;

    let change = current_brightness - last_brightness;
    let abs_change = change.abs();

    if is_dimming {
        // Check for significant dimming (negative change)
        change < 0.0 && abs_change > DIMMING_THRESHOLD
    } else {
        // Check for significant brightening (positive change)
        change > 0.0 && abs_change > BRIGHTENING_THRESHOLD
    }
}

/// Branchless significant change detection.
///
/// Uses bit manipulation to avoid branches entirely,
/// which can be faster on some architectures.
#[inline]
pub fn check_significant_change_branchless(
    current_brightness: f32,
    last_brightness: f32,
    is_dimming: bool,
) -> bool {
    const DIMMING_THRESHOLD: f32 = 8.0;
    const BRIGHTENING_THRESHOLD: f32 = 12.0;

    let change = current_brightness - last_brightness;
    let abs_change = change.abs();

    // Convert conditions to 0/1
    let is_negative = (change < 0.0) as u32;
    let is_positive = (change > 0.0) as u32;
    let exceeds_dim = (abs_change > DIMMING_THRESHOLD) as u32;
    let exceeds_bright = (abs_change > BRIGHTENING_THRESHOLD) as u32;
    let dimming_flag = is_dimming as u32;

    // Combine conditions branchlessly
    let dimming_result = dimming_flag & is_negative & exceeds_dim;
    let brightening_result = (1 - dimming_flag) & is_positive & exceeds_bright;

    (dimming_result | brightening_result) != 0
}

/// Check any significant change regardless of direction.
///
/// Useful for triggering any response to environmental changes.
#[inline]
pub fn check_any_significant_change(
    current_brightness: f32,
    last_brightness: f32,
) -> bool {
    const THRESHOLD: f32 = 10.0;
    (current_brightness - last_brightness).abs() > THRESHOLD
}

/// Determine change direction and magnitude.
///
/// Returns (is_significant, is_dimming, magnitude)
#[inline]
pub fn analyze_change(
    current_brightness: f32,
    last_brightness: f32,
) -> (bool, bool, f32) {
    const DIMMING_THRESHOLD: f32 = 8.0;
    const BRIGHTENING_THRESHOLD: f32 = 12.0;

    let change = current_brightness - last_brightness;
    let abs_change = change.abs();
    let is_dimming = change < 0.0;

    let threshold = if is_dimming {
        DIMMING_THRESHOLD
    } else {
        BRIGHTENING_THRESHOLD
    };

    let is_significant = abs_change > threshold;

    (is_significant, is_dimming, abs_change)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_significant_dimming() {
        // 50 -> 30 is a -20 change, exceeds 8 threshold
        assert!(check_significant_change(30.0, 50.0, true));
    }

    #[test]
    fn test_small_dimming() {
        // 50 -> 45 is only -5, below 8 threshold
        assert!(!check_significant_change(45.0, 50.0, true));
    }

    #[test]
    fn test_significant_brightening() {
        // 30 -> 50 is +20, exceeds 12 threshold
        assert!(check_significant_change(50.0, 30.0, false));
    }

    #[test]
    fn test_small_brightening() {
        // 30 -> 40 is only +10, below 12 threshold
        assert!(!check_significant_change(40.0, 30.0, false));
    }

    #[test]
    fn test_dimming_when_checking_brightening() {
        // Light is dimming but we're checking for brightening
        assert!(!check_significant_change(30.0, 50.0, false));
    }

    #[test]
    fn test_brightening_when_checking_dimming() {
        // Light is brightening but we're checking for dimming
        assert!(!check_significant_change(50.0, 30.0, true));
    }

    #[test]
    fn test_branchless_matches_branched() {
        for current in (0..=100).step_by(5) {
            for last in (0..=100).step_by(5) {
                for is_dimming in [true, false] {
                    let branched = check_significant_change(
                        current as f32, last as f32, is_dimming
                    );
                    let branchless = check_significant_change_branchless(
                        current as f32, last as f32, is_dimming
                    );
                    assert_eq!(
                        branched, branchless,
                        "Mismatch: current={}, last={}, is_dimming={}",
                        current, last, is_dimming
                    );
                }
            }
        }
    }

    #[test]
    fn test_any_change() {
        assert!(check_any_significant_change(30.0, 50.0));
        assert!(check_any_significant_change(60.0, 45.0));
        assert!(!check_any_significant_change(48.0, 50.0));
    }

    #[test]
    fn test_analyze_change() {
        let (sig, dim, mag) = analyze_change(30.0, 50.0);
        assert!(sig);
        assert!(dim);
        assert!((mag - 20.0).abs() < 0.1);

        let (sig, dim, mag) = analyze_change(45.0, 50.0);
        assert!(!sig);
        assert!(dim);
        assert!((mag - 5.0).abs() < 0.1);
    }
}
