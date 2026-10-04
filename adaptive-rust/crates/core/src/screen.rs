//! Screen content brightness analysis
//!
//! Analyzes screen content to adjust brightness based on
//! what's currently displayed (dark IDE vs bright browser).

/// Analyze screen brightness and return adjustment factor.
///
/// # Arguments
/// * `pixels` - Screen pixel data (grayscale or will compute luminance)
///
/// # Returns
/// Adjustment factor:
/// - `> 1.0` for bright content (increase brightness for visibility)
/// - `< 1.0` for dark content (decrease brightness for comfort)
/// - `1.0` for neutral content
///
/// # Performance
/// - SIMD-optimized mean calculation
/// - Target: 0.02ms for 960x540 sample (vs 0.08ms Numba)
#[inline]
pub fn analyze_screen_brightness(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 1.0;
    }

    // Calculate mean brightness using SIMD-friendly loop
    let len = pixels.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let mut sum: u64 = 0;
    let mut i = 0;

    // Process 16 bytes at a time
    for _ in 0..chunks {
        let mut chunk_sum: u64 = 0;
        for j in 0..16 {
            chunk_sum += unsafe { *pixels.get_unchecked(i + j) as u64 };
        }
        sum += chunk_sum;
        i += 16;
    }

    // Handle remainder
    for j in 0..remainder {
        sum += unsafe { *pixels.get_unchecked(i + j) as u64 };
    }

    let mean = sum as f32 / len as f32;
    let brightness = mean / 255.0;

    // Return adjustment factor based on content brightness
    if brightness > 0.7 {
        // Very bright content (white pages, bright websites)
        1.2
    } else if brightness < 0.3 {
        // Dark content (dark IDE themes, movies)
        0.8
    } else {
        // Neutral content
        1.0
    }
}

/// Analyze screen with RGB data (3 bytes per pixel).
///
/// Computes luminance using standard formula:
/// Y = 0.299*R + 0.587*G + 0.114*B
#[inline]
pub fn analyze_screen_brightness_rgb(pixels: &[u8]) -> f32 {
    if pixels.len() < 3 {
        return 1.0;
    }

    let pixel_count = pixels.len() / 3;
    let mut sum: f64 = 0.0;

    for i in 0..pixel_count {
        let r = unsafe { *pixels.get_unchecked(i * 3) as f64 };
        let g = unsafe { *pixels.get_unchecked(i * 3 + 1) as f64 };
        let b = unsafe { *pixels.get_unchecked(i * 3 + 2) as f64 };

        // Standard luminance formula
        let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        sum += luminance;
    }

    let mean = sum / pixel_count as f64;
    let brightness = mean / 255.0;

    if brightness > 0.7 {
        1.2
    } else if brightness < 0.3 {
        0.8
    } else {
        1.0
    }
}

/// Analyze screen with RGBA data (4 bytes per pixel).
#[inline]
pub fn analyze_screen_brightness_rgba(pixels: &[u8]) -> f32 {
    if pixels.len() < 4 {
        return 1.0;
    }

    let pixel_count = pixels.len() / 4;
    let mut sum: f64 = 0.0;

    for i in 0..pixel_count {
        let r = unsafe { *pixels.get_unchecked(i * 4) as f64 };
        let g = unsafe { *pixels.get_unchecked(i * 4 + 1) as f64 };
        let b = unsafe { *pixels.get_unchecked(i * 4 + 2) as f64 };
        // Alpha channel at i*4+3 is ignored

        let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        sum += luminance;
    }

    let mean = sum / pixel_count as f64;
    let brightness = mean / 255.0;

    if brightness > 0.7 {
        1.2
    } else if brightness < 0.3 {
        0.8
    } else {
        1.0
    }
}

/// Continuous adjustment factor instead of discrete thresholds.
///
/// Returns smooth interpolation between 0.8 and 1.2 based on brightness.
#[inline]
pub fn analyze_screen_brightness_smooth(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 1.0;
    }

    let len = pixels.len();
    let sum: u64 = pixels.iter().map(|&x| x as u64).sum();
    let mean = sum as f32 / len as f32;
    let brightness = mean / 255.0;

    // Smooth interpolation: 0.0 -> 0.8, 0.5 -> 1.0, 1.0 -> 1.2
    0.8 + brightness * 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(analyze_screen_brightness(&[]), 1.0);
    }

    #[test]
    fn test_bright_content() {
        let pixels = vec![200u8; 1000];
        let factor = analyze_screen_brightness(&pixels);
        assert_eq!(factor, 1.2);
    }

    #[test]
    fn test_dark_content() {
        let pixels = vec![50u8; 1000];
        let factor = analyze_screen_brightness(&pixels);
        assert_eq!(factor, 0.8);
    }

    #[test]
    fn test_neutral_content() {
        let pixels = vec![128u8; 1000];
        let factor = analyze_screen_brightness(&pixels);
        assert_eq!(factor, 1.0);
    }

    #[test]
    fn test_rgb_bright() {
        // White pixels (255, 255, 255)
        let pixels = vec![255u8; 3000]; // 1000 RGB pixels
        let factor = analyze_screen_brightness_rgb(&pixels);
        assert_eq!(factor, 1.2);
    }

    #[test]
    fn test_rgba_dark() {
        // Dark pixels (30, 30, 30, 255)
        let mut pixels = Vec::with_capacity(4000);
        for _ in 0..1000 {
            pixels.extend_from_slice(&[30, 30, 30, 255]);
        }
        let factor = analyze_screen_brightness_rgba(&pixels);
        assert_eq!(factor, 0.8);
    }

    #[test]
    fn test_smooth_interpolation() {
        let dark = vec![0u8; 100];
        let bright = vec![255u8; 100];
        let mid = vec![128u8; 100];

        let dark_factor = analyze_screen_brightness_smooth(&dark);
        let bright_factor = analyze_screen_brightness_smooth(&bright);
        let mid_factor = analyze_screen_brightness_smooth(&mid);

        assert!((dark_factor - 0.8).abs() < 0.01);
        assert!((bright_factor - 1.2).abs() < 0.01);
        assert!((mid_factor - 1.0).abs() < 0.05);
    }
}
