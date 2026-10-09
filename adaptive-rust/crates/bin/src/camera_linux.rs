//! Real Linux V4L2 capture restored from 5094193; reject compressed/non-YUYV frames.
use anyhow::{Context, Result};
use std::time::Duration;
use tracing::info;
use v4l::{buffer::Type, io::traits::CaptureStream, prelude::*, video::Capture, FourCC};

pub struct Camera {
    stream: MmapStream<'static>,
    width: usize,
    height: usize,
    stride: usize,
}

impl Camera {
    pub fn new(index: usize) -> Result<Self> {
        let path = format!("/dev/video{index}");
        let dev = Device::with_path(&path).with_context(|| format!("Failed to open {path}"))?;
        let mut fmt = dev.format()?;
        fmt.width = 320;
        fmt.height = 240;
        fmt.fourcc = FourCC::new(b"YUYV");
        let actual = dev.set_format(&fmt)?;
        anyhow::ensure!(
            actual.fourcc == FourCC::new(b"YUYV"),
            "Camera did not negotiate YUYV: {:?}",
            actual.fourcc
        );
        let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)?;
        stream.set_timeout(Duration::from_secs(1));
        info!(
            "Real V4L2 camera {path}: {}x{} YUYV",
            actual.width, actual.height
        );
        Ok(Self {
            stream,
            width: actual.width as usize,
            height: actual.height as usize,
            stride: actual.stride as usize,
        })
    }

    pub fn capture_frame(&mut self) -> Result<Vec<u8>> {
        let (buf, metadata) = self.stream.next()?;
        let used = metadata.bytesused as usize;
        anyhow::ensure!(used <= buf.len(), "Invalid camera bytesused");
        decode_yuyv(&buf[..used], self.width, self.height, self.stride)
    }
}

fn decode_yuyv(buf: &[u8], width: usize, height: usize, stride: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        width > 0 && height > 0 && width % 2 == 0,
        "Invalid YUYV dimensions"
    );
    let row = width.checked_mul(2).context("YUYV row overflow")?;
    anyhow::ensure!(stride >= row, "Invalid YUYV stride");
    let size = stride.checked_mul(height).context("YUYV size overflow")?;
    anyhow::ensure!(buf.len() >= size, "Truncated YUYV frame");
    Ok(buf[..size]
        .chunks_exact(stride)
        .flat_map(|r| r[..row].iter().step_by(2).copied())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn luminance_ignores_chroma_and_padding() {
        assert_eq!(
            decode_yuyv(
                &[10, 200, 20, 220, 99, 99, 30, 201, 40, 221, 88, 88],
                2,
                2,
                6
            )
            .unwrap(),
            vec![10, 20, 30, 40]
        );
    }
    #[test]
    fn malformed_or_empty_frames_are_refused() {
        for (data, w, h, s) in [
            (&[][..], 0, 1, 0),
            (&[1, 2, 3][..], 2, 1, 4),
            (&[0; 4][..], 2, 1, 3),
            (&[0; 6][..], 3, 1, 6),
        ] {
            assert!(decode_yuyv(data, w, h, s).is_err());
        }
    }
}
