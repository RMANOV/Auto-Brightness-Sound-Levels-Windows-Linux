//! PyO3 Python bindings for adaptive-core
//!
//! Provides zero-copy NumPy interop for maximum performance.
//! Build with: maturin develop --release

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute RMS noise level from audio samples.
///
/// Args:
///     audio: numpy.ndarray[float32] - Audio samples
///
/// Returns:
///     float: RMS noise level
#[pyfunction]
fn compute_noise_level(audio: PyReadonlyArray1<f32>) -> f32 {
    adaptive_core::compute_noise_level(audio.as_slice().unwrap())
}

/// Calculate brightness from grayscale frame.
///
/// Args:
///     frame: numpy.ndarray[uint8] - Grayscale pixel data
///
/// Returns:
///     float: Brightness percentage (0-100)
#[pyfunction]
fn calculate_brightness(frame: PyReadonlyArray1<u8>) -> f32 {
    adaptive_core::calculate_brightness(frame.as_slice().unwrap())
}

/// Map camera brightness to screen brightness with boost curve.
///
/// Args:
///     camera_brightness: float - Camera detected brightness (0-100)
///     min_brightness: float - Minimum screen brightness
///     max_brightness: float - Maximum screen brightness
///
/// Returns:
///     float: Target screen brightness
#[pyfunction]
fn calculate_brightness_mapping(
    camera_brightness: f32,
    min_brightness: f32,
    max_brightness: f32,
) -> f32 {
    adaptive_core::calculate_brightness_mapping(camera_brightness, min_brightness, max_brightness)
}

/// Map noise level to volume with logarithmic curve.
///
/// Args:
///     normalized_noise: float - Normalized noise level (0-1)
///     min_volume: float - Minimum volume percentage
///     max_volume: float - Maximum volume percentage
///
/// Returns:
///     float: Target volume percentage
#[pyfunction]
fn calculate_volume_mapping(normalized_noise: f32, min_volume: f32, max_volume: f32) -> f32 {
    adaptive_core::calculate_volume_mapping(normalized_noise, min_volume, max_volume)
}

/// Smooth transition between values.
///
/// Args:
///     current: float - Current value
///     target: float - Target value
///     smoothing_factor: float - Smoothing factor (0-1)
///
/// Returns:
///     float: Smoothed value
#[pyfunction]
fn smooth_transition(current: f32, target: f32, smoothing_factor: f32) -> f32 {
    adaptive_core::smooth_transition(current, target, smoothing_factor)
}

/// Analyze screen content brightness.
///
/// Args:
///     pixels: numpy.ndarray[uint8] - Screen pixel data (grayscale)
///
/// Returns:
///     float: Brightness adjustment factor (0.8-1.2)
#[pyfunction]
fn analyze_screen_brightness(pixels: PyReadonlyArray1<u8>) -> f32 {
    adaptive_core::analyze_screen_brightness(pixels.as_slice().unwrap())
}

/// Check for significant brightness change.
///
/// Args:
///     current_brightness: float - Current brightness
///     last_brightness: float - Previous brightness
///     is_dimming: bool - Whether checking for dimming
///
/// Returns:
///     bool: True if change is significant
#[pyfunction]
fn check_significant_change(current_brightness: f32, last_brightness: f32, is_dimming: bool) -> bool {
    adaptive_core::check_significant_change(current_brightness, last_brightness, is_dimming)
}

/// Batch compute noise levels for multiple audio segments.
///
/// Args:
///     segments: List of numpy.ndarray[float32]
///
/// Returns:
///     numpy.ndarray[float32]: RMS values for each segment
#[pyfunction]
fn compute_noise_levels_batch<'py>(
    py: Python<'py>,
    segments: Vec<PyReadonlyArray1<f32>>,
) -> Bound<'py, PyArray1<f32>> {
    let results: Vec<f32> = segments
        .iter()
        .map(|seg| adaptive_core::compute_noise_level(seg.as_slice().unwrap()))
        .collect();
    PyArray1::from_vec(py, results)
}

/// Module version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition
#[pymodule]
fn adaptive_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_noise_level, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_brightness, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_brightness_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_volume_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_transition, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_screen_brightness, m)?)?;
    m.add_function(wrap_pyfunction!(check_significant_change, m)?)?;
    m.add_function(wrap_pyfunction!(compute_noise_levels_batch, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
