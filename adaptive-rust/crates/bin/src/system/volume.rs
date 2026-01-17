//! Volume control for Linux
//!
//! Supports amixer, pactl, and wpctl.

use anyhow::{Context, Result};
use std::process::Command;
use tracing::{debug, info, warn};

/// Volume control method
#[derive(Debug, Clone)]
enum Method {
    Amixer,
    Pactl,
    Wpctl,
}

/// Volume controller
pub struct VolumeControl {
    method: Method,
}

impl VolumeControl {
    /// Detect and initialize volume control
    pub fn new() -> Result<Self> {
        // Try amixer first (ALSA)
        if Self::has_command("amixer") {
            info!("Using amixer for volume control");
            return Ok(Self { method: Method::Amixer });
        }

        // Try pactl (PulseAudio)
        if Self::has_command("pactl") {
            info!("Using pactl for volume control");
            return Ok(Self { method: Method::Pactl });
        }

        // Try wpctl (PipeWire)
        if Self::has_command("wpctl") {
            info!("Using wpctl for volume control");
            return Ok(Self { method: Method::Wpctl });
        }

        anyhow::bail!("No volume control method available")
    }

    fn has_command(cmd: &str) -> bool {
        Command::new("which")
            .arg(cmd)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get current volume (0-100)
    pub fn get(&self) -> Result<i32> {
        match &self.method {
            Method::Amixer => {
                let output = Command::new("amixer")
                    .arg("get")
                    .arg("Master")
                    .output()
                    .context("Failed to run amixer")?;

                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse output like: [75%]
                for line in stdout.lines() {
                    if let Some(start) = line.find('[') {
                        if let Some(end) = line[start..].find('%') {
                            let percent_str = &line[start + 1..start + end];
                            if let Ok(percent) = percent_str.parse::<i32>() {
                                return Ok(percent);
                            }
                        }
                    }
                }

                anyhow::bail!("Could not parse amixer output")
            }

            Method::Pactl => {
                let output = Command::new("pactl")
                    .args(["get-sink-volume", "@DEFAULT_SINK@"])
                    .output()
                    .context("Failed to run pactl")?;

                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse output like: Volume: front-left: 65536 / 100% / 0.00 dB
                for part in stdout.split_whitespace() {
                    if part.ends_with('%') {
                        if let Ok(percent) = part.trim_end_matches('%').parse::<i32>() {
                            return Ok(percent);
                        }
                    }
                }

                anyhow::bail!("Could not parse pactl output")
            }

            Method::Wpctl => {
                let output = Command::new("wpctl")
                    .args(["get-volume", "@DEFAULT_AUDIO_SINK@"])
                    .output()
                    .context("Failed to run wpctl")?;

                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse output like: Volume: 0.75
                if let Some(volume_str) = stdout.split_whitespace().nth(1) {
                    if let Ok(volume) = volume_str.parse::<f32>() {
                        return Ok((volume * 100.0) as i32);
                    }
                }

                anyhow::bail!("Could not parse wpctl output")
            }
        }
    }

    /// Set volume (0-100)
    pub fn set(&self, percent: i32) -> Result<()> {
        let percent = percent.clamp(0, 100);
        debug!("Setting volume to {}%", percent);

        match &self.method {
            Method::Amixer => {
                let status = Command::new("amixer")
                    .args(["set", "Master", &format!("{}%", percent)])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .context("Failed to run amixer")?;

                if !status.success() {
                    warn!("Failed to set volume with amixer");
                }
            }

            Method::Pactl => {
                let status = Command::new("pactl")
                    .args(["set-sink-volume", "@DEFAULT_SINK@", &format!("{}%", percent)])
                    .status()
                    .context("Failed to run pactl")?;

                if !status.success() {
                    warn!("Failed to set volume with pactl");
                }
            }

            Method::Wpctl => {
                let volume = percent as f32 / 100.0;
                let status = Command::new("wpctl")
                    .args(["set-volume", "@DEFAULT_AUDIO_SINK@", &volume.to_string()])
                    .status()
                    .context("Failed to run wpctl")?;

                if !status.success() {
                    warn!("Failed to set volume with wpctl");
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_command() {
        assert!(VolumeControl::has_command("ls"));
        assert!(!VolumeControl::has_command("nonexistent_command_xyz"));
    }
}
