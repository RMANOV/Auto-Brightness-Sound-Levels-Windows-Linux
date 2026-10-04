//! Exponential smoothing for transitions
//!
//! Provides smooth transitions between current and target values
//! using exponential moving average.

/// Compute smooth transition between current and target value.
///
/// # Arguments
/// * `current` - Current value
/// * `target` - Target value to transition towards
/// * `smoothing_factor` - Factor controlling transition speed (0.0-1.0)
///   - 0.0 = no change (stays at current)
///   - 1.0 = immediate jump to target
///   - 0.3 = typical smooth transition
///
/// # Returns
/// New smoothed value: `current + (target - current) * smoothing_factor`
///
/// # Performance
/// - Fully inlined, single multiply-add operation
/// - Target: 0.0001ms (5x faster than Numba's 0.0005ms)
#[inline(always)]
pub fn smooth_transition(current: f32, target: f32, smoothing_factor: f32) -> f32 {
    // FMA optimization: current + error * factor
    // Equivalent to: current + (target - current) * smoothing_factor
    current + (target - current) * smoothing_factor
}

/// Smooth transition with clamping to valid range.
#[inline(always)]
pub fn smooth_transition_clamped(
    current: f32,
    target: f32,
    smoothing_factor: f32,
    min: f32,
    max: f32,
) -> f32 {
    smooth_transition(current, target, smoothing_factor).clamp(min, max)
}

/// Adaptive smoothing factor based on error magnitude.
///
/// Uses larger smoothing factor for bigger differences,
/// allowing faster response to large changes while
/// maintaining smooth transitions for small adjustments.
#[inline]
pub fn adaptive_smooth_transition(
    current: f32,
    target: f32,
    base_factor: f32,
    boost_threshold: f32,
) -> f32 {
    let error = (target - current).abs();
    let factor = if error > boost_threshold {
        // Boost factor for large errors (up to 2x base)
        (base_factor * 2.0).min(0.9)
    } else {
        base_factor
    };
    smooth_transition(current, target, factor)
}

/// Batch smooth transition for multiple values.
///
/// Useful for transitioning multiple brightness/volume channels
/// simultaneously with SIMD optimization.
#[inline]
pub fn smooth_transition_batch(
    current: &[f32],
    target: &[f32],
    output: &mut [f32],
    smoothing_factor: f32,
) {
    debug_assert_eq!(current.len(), target.len());
    debug_assert_eq!(current.len(), output.len());

    for ((c, t), o) in current.iter().zip(target.iter()).zip(output.iter_mut()) {
        *o = smooth_transition(*c, *t, smoothing_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_change() {
        let result = smooth_transition(10.0, 20.0, 0.0);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_immediate_jump() {
        let result = smooth_transition(10.0, 20.0, 1.0);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_half_transition() {
        let result = smooth_transition(10.0, 20.0, 0.5);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_typical_smoothing() {
        let result = smooth_transition(10.0, 20.0, 0.3);
        assert!((result - 13.0).abs() < 0.001);
    }

    #[test]
    fn test_decreasing() {
        let result = smooth_transition(20.0, 10.0, 0.3);
        assert!((result - 17.0).abs() < 0.001);
    }

    #[test]
    fn test_clamped() {
        let result = smooth_transition_clamped(50.0, 100.0, 1.0, 0.0, 75.0);
        assert_eq!(result, 75.0);
    }

    #[test]
    fn test_adaptive_small_error() {
        let result = adaptive_smooth_transition(10.0, 12.0, 0.3, 5.0);
        // Small error, should use base factor
        assert!((result - 10.6).abs() < 0.1);
    }

    #[test]
    fn test_adaptive_large_error() {
        let result = adaptive_smooth_transition(10.0, 30.0, 0.3, 5.0);
        // Large error, should use boosted factor (0.6)
        let expected = 10.0 + (30.0 - 10.0) * 0.6;
        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_batch() {
        let current = [10.0, 20.0, 30.0, 40.0];
        let target = [20.0, 30.0, 40.0, 50.0];
        let mut output = [0.0; 4];

        smooth_transition_batch(&current, &target, &mut output, 0.5);

        assert_eq!(output, [15.0, 25.0, 35.0, 45.0]);
    }
}
