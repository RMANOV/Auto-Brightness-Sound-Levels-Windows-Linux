//! Volume mapping with logarithmic curve
//!
//! Maps normalized noise level to volume percentage using
//! a perceptually-accurate logarithmic curve.

/// Map normalized noise level to volume with logarithmic curve.
///
/// # Arguments
/// * `normalized_noise` - Noise level normalized to 0.0-1.0 range
/// * `min_volume` - Minimum volume percentage
/// * `max_volume` - Maximum volume percentage
///
/// # Returns
/// Target volume percentage
///
/// # Algorithm
/// Enhanced logarithmic curve with parameters:
/// - `curve_factor = 0.55` - Controls curve steepness
/// - `multiplier = 12.0` - Scales noise before log
/// - `bias = 0.22` - Baseline offset (ensures min audible level)
///
/// # Performance
/// - Uses the standard f32 logarithm; no approximation or measured speed claim.
#[inline]
pub fn calculate_volume_mapping(normalized_noise: f32, min_volume: f32, max_volume: f32) -> f32 {
    const CURVE_FACTOR: f32 = 0.55;
    const MULTIPLIER: f32 = 12.0;
    const BIAS: f32 = 0.22;

    let adjusted_noise = if normalized_noise > 0.0 {
        // Enhanced curve: noise^0.8 * 1.2 for better response
        let enhanced = (normalized_noise.powf(0.8) * 1.2).min(1.0);

        // Logarithmic mapping
        let log_result = CURVE_FACTOR * (1.0 + MULTIPLIER * enhanced).log10() + BIAS;
        log_result.clamp(0.0, 1.0)
    } else {
        BIAS
    };

    let volume_range = max_volume - min_volume;
    adjusted_noise * volume_range + min_volume
}

/// Compatibility entry point for log10, using the standard implementation.
///
/// The former two-term mantissa approximation exceeded the volume mapping's
/// accuracy contract and mishandled subnormal and nonpositive inputs.
#[inline]
pub fn fast_log10(x: f32) -> f32 {
    x.log10()
}

/// Compatibility volume mapping with the same logarithmic curve and accuracy.
#[inline]
pub fn calculate_volume_mapping_fast(
    normalized_noise: f32,
    min_volume: f32,
    max_volume: f32,
) -> f32 {
    const CURVE_FACTOR: f32 = 0.55;
    const MULTIPLIER: f32 = 12.0;
    const BIAS: f32 = 0.22;

    let adjusted_noise = if normalized_noise > 0.0 {
        let enhanced = (normalized_noise.powf(0.8) * 1.2).min(1.0);
        let log_result = CURVE_FACTOR * fast_log10(1.0 + MULTIPLIER * enhanced) + BIAS;
        log_result.clamp(0.0, 1.0)
    } else {
        BIAS
    };

    let volume_range = max_volume - min_volume;
    adjusted_noise * volume_range + min_volume
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_noise() {
        let result = calculate_volume_mapping(0.0, 3.0, 35.0);
        // With bias 0.22, volume should be 3 + 0.22 * 32 = 10.04
        assert!((result - 10.04).abs() < 0.5);
    }

    #[test]
    fn test_max_noise() {
        let result = calculate_volume_mapping(1.0, 3.0, 35.0);
        // Should be close to max
        assert!(result > 25.0 && result <= 35.0);
    }

    #[test]
    fn test_mid_noise() {
        let result = calculate_volume_mapping(0.5, 3.0, 35.0);
        // Should be between min and max
        assert!(result > 3.0 && result < 35.0);
    }

    #[test]
    fn test_fast_log10_accuracy() {
        for i in 1..100 {
            let x = i as f32 * 0.1;
            let std_log = x.log10();
            let fast = fast_log10(x);
            let error = (std_log - fast).abs() / std_log.abs().max(0.001);
            assert!(
                error < 0.02,
                "Error too high for x={}: std={}, fast={}",
                x,
                std_log,
                fast
            );
        }
    }

    #[test]
    fn test_fast_matches_standard() {
        for i in 0..=100 {
            let noise = i as f32 / 100.0;
            let std_result = calculate_volume_mapping(noise, 3.0, 35.0);
            let fast_result = calculate_volume_mapping_fast(noise, 3.0, 35.0);
            assert!(
                (std_result - fast_result).abs() < 0.5,
                "Mismatch at noise={}: std={}, fast={}",
                noise,
                std_result,
                fast_result
            );
        }
    }
    #[test]
    fn logarithm_special_values_follow_ieee_contract() {
        assert_eq!(fast_log10(0.0), f32::NEG_INFINITY);
        assert!(fast_log10(-1.0).is_nan());
        assert!(fast_log10(f32::NAN).is_nan());
        assert_eq!(fast_log10(f32::INFINITY), f32::INFINITY);
    }
    #[test]
    fn logarithm_handles_subnormal_and_power_boundaries() {
        for x in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            0.9999999,
            1.0,
            1.0000001,
            1.9999999,
            2.0,
            2.0000002,
            f32::MAX,
        ] {
            let reference = (x as f64).log10();
            assert!((fast_log10(x) as f64 - reference).abs() < 0.00001, "x={x}");
        }
    }
    #[test]
    fn fast_volume_agrees_across_dense_supported_noise_domain() {
        for i in 0..=10000 {
            let noise = i as f32 / 10000.0;
            assert!(
                (calculate_volume_mapping_fast(noise, 3.0, 35.0)
                    - calculate_volume_mapping(noise, 3.0, 35.0))
                .abs()
                    < 0.5,
                "noise={noise}"
            );
        }
    }
}
