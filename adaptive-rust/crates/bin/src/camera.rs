//! Camera capture using v4l2
//!
//! Captures frames from webcam for ambient light detection.

use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer};
use std::io;
use tracing::{debug, info};
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

/// Camera wrapper for v4l2 capture
pub struct Camera {
    stream: MmapStream<'static>,
    width: u32,
    height: u32,
}

impl Camera {
    /// Open camera at given index
    pub fn new(index: usize) -> Result<Self> {
        let path = format!("/dev/video{}", index);
        info!("Opening camera at {}", path);

        let dev = Device::with_path(&path)
            .with_context(|| format!("Failed to open camera at {}", path))?;

        // Set format - prefer YUYV or MJPEG at 320x240 for performance
        let mut fmt = dev.format()?;
        fmt.width = 320;
        fmt.height = 240;

        // Try YUYV first (easy to convert to grayscale)
        fmt.fourcc = FourCC::new(b"YUYV");
        if dev.set_format(&fmt).is_err() {
            // Fall back to MJPEG
            fmt.fourcc = FourCC::new(b"MJPG");
            dev.set_format(&fmt)?;
        }

        let actual_fmt = dev.format()?;
        info!("Camera format: {}x{} {:?}",
            actual_fmt.width, actual_fmt.height, actual_fmt.fourcc);

        // Create memory-mapped stream
        let stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)?;

        Ok(Self {
            stream,
            width: actual_fmt.width,
            height: actual_fmt.height,
        })
    }

    /// Capture a frame and return grayscale pixel data
    pub fn capture_frame(&self) -> Result<Vec<u8>> {
        let (buf, _meta) = self.stream.next()?;

        // Convert to grayscale based on format
        // For YUYV, every other byte is Y (luminance)
        let grayscale: Vec<u8> = buf.iter()
            .step_by(2)
            .copied()
            .collect();

        debug!("Captured frame: {} bytes -> {} grayscale pixels",
            buf.len(), grayscale.len());

        Ok(grayscale)
    }

    /// Get frame dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Fallback camera using OpenCV-style capture
/// (useful if v4l2 doesn't work)
pub struct FallbackCamera {
    width: u32,
    height: u32,
}

impl FallbackCamera {
    pub fn new(_index: usize) -> Result<Self> {
        // This would use opencv crate or similar
        // For now, return simulated data
        Ok(Self {
            width: 320,
            height: 240,
        })
    }

    pub fn capture_frame(&self) -> Result<Vec<u8>> {
        // Return neutral gray frame for testing
        Ok(vec![128u8; (self.width * self.height) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_camera() {
        let cam = FallbackCamera::new(0).unwrap();
        let frame = cam.capture_frame().unwrap();
        assert_eq!(frame.len(), 320 * 240);
    }
}
