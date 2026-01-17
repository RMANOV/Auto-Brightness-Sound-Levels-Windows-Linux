//! SIMD-optimized audio processing
//!
//! RMS (Root Mean Square) calculation for ambient noise level detection.
//! Target: 0.03ms (5x faster than Numba's 0.15ms)

/// Compute RMS noise level from audio samples using SIMD optimization.
///
/// # Arguments
/// * `audio` - Slice of f32 audio samples (typically 4410 samples for 100ms at 44.1kHz)
///
/// # Returns
/// RMS value representing ambient noise level (typically 1e-6 to 1e-2 range)
///
/// # Performance
/// - Uses manual loop unrolling for SIMD auto-vectorization
/// - Processes 8 samples per iteration when possible
/// - Falls back to scalar for remainder
#[inline]
pub fn compute_noise_level(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    let len = audio.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;
    let mut sum4: f32 = 0.0;
    let mut sum5: f32 = 0.0;
    let mut sum6: f32 = 0.0;
    let mut sum7: f32 = 0.0;

    // Process 8 elements at a time for SIMD vectorization
    let mut i = 0;
    for _ in 0..chunks {
        let v0 = unsafe { *audio.get_unchecked(i) };
        let v1 = unsafe { *audio.get_unchecked(i + 1) };
        let v2 = unsafe { *audio.get_unchecked(i + 2) };
        let v3 = unsafe { *audio.get_unchecked(i + 3) };
        let v4 = unsafe { *audio.get_unchecked(i + 4) };
        let v5 = unsafe { *audio.get_unchecked(i + 5) };
        let v6 = unsafe { *audio.get_unchecked(i + 6) };
        let v7 = unsafe { *audio.get_unchecked(i + 7) };

        sum0 += v0 * v0;
        sum1 += v1 * v1;
        sum2 += v2 * v2;
        sum3 += v3 * v3;
        sum4 += v4 * v4;
        sum5 += v5 * v5;
        sum6 += v6 * v6;
        sum7 += v7 * v7;

        i += 8;
    }

    // Handle remainder
    let mut sum_remainder: f32 = 0.0;
    for j in 0..remainder {
        let v = unsafe { *audio.get_unchecked(i + j) };
        sum_remainder += v * v;
    }

    // Combine all partial sums
    let total_sum = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7) + sum_remainder;

    (total_sum / len as f32).sqrt()
}

/// SIMD-optimized RMS using Rayon for very large buffers
///
/// Use this for audio buffers > 100K samples where parallelization overhead
/// is amortized.
#[inline]
pub fn compute_noise_level_parallel(audio: &[f32]) -> f32 {
    use rayon::prelude::*;

    if audio.is_empty() {
        return 0.0;
    }

    // Only parallelize for large buffers (>100K samples)
    if audio.len() < 100_000 {
        return compute_noise_level(audio);
    }

    let sum: f32 = audio.par_chunks(8192).map(|chunk| {
        chunk.iter().map(|&x| x * x).sum::<f32>()
    }).sum();

    (sum / audio.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio() {
        assert_eq!(compute_noise_level(&[]), 0.0);
    }

    #[test]
    fn test_silent_audio() {
        let audio = vec![0.0; 4410];
        assert_eq!(compute_noise_level(&audio), 0.0);
    }

    #[test]
    fn test_constant_signal() {
        let audio = vec![0.5; 4410];
        let rms = compute_noise_level(&audio);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sine_wave() {
        // RMS of sine wave = amplitude / sqrt(2)
        let audio: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 44.0).sin())
            .collect();
        let rms = compute_noise_level(&audio);
        // Expected: 1.0 / sqrt(2) ≈ 0.707
        assert!((rms - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let audio: Vec<f32> = (0..200_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let seq = compute_noise_level(&audio);
        let par = compute_noise_level_parallel(&audio);
        assert!((seq - par).abs() < 0.0001);
    }
}
