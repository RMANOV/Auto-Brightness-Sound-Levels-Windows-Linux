//! Audio capture using cpal (cross-platform)
//!
//! Captures audio samples for ambient noise level detection.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Audio capture wrapper
pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
}

impl AudioCapture {
    /// Initialize audio capture with default input device
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .context("No input device available")?;

        info!("Using audio device: {}", device.name().unwrap_or_default());

        let config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        info!(
            "Audio config: {} Hz, {} channels",
            config.sample_rate().0,
            config.channels()
        );

        let stream_config: StreamConfig = config.into();

        Ok(Self {
            device,
            config: stream_config,
        })
    }

    /// Capture audio samples for specified duration
    pub fn capture_samples(&self, duration: Duration) -> Result<Vec<f32>> {
        let sample_rate = self.config.sample_rate.0 as usize;
        let channels = self.config.channels as usize;
        let expected_samples = (sample_rate * duration.as_millis() as usize / 1000) * channels;

        let samples: Arc<Mutex<Vec<f32>>> =
            Arc::new(Mutex::new(Vec::with_capacity(expected_samples)));
        let samples_clone = Arc::clone(&samples);

        let err_fn = |err| warn!("Audio stream error: {}", err);

        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut samples = samples_clone.lock();
                samples.extend_from_slice(data);
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        std::thread::sleep(duration);
        drop(stream);

        let samples = Arc::try_unwrap(samples)
            .map_err(|_| anyhow::anyhow!("Failed to unwrap samples"))?
            .into_inner();

        // Convert to mono by averaging channels
        let mono: Vec<f32> = if channels > 1 {
            samples
                .chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        debug!("Captured {} mono samples", mono.len());

        Ok(mono)
    }
}
