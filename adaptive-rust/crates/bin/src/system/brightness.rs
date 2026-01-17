//! Brightness control for Linux
//!
//! Supports brightnessctl, xbacklight, and direct sysfs access.

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tracing::{debug, info, warn};

/// Brightness control method
#[derive(Debug, Clone)]
enum Method {
    Brightnessctl,
    Xbacklight,
    Sysfs(PathBuf),
}

/// Brightness controller
pub struct BrightnessControl {
    method: Method,
    max_brightness: Option<i32>,
}

impl BrightnessControl {
    /// Detect and initialize brightness control
    pub fn new() -> Result<Self> {
        // Try brightnessctl first
        if Self::has_command("brightnessctl") {
            info!("Using brightnessctl for brightness control");
            return Ok(Self {
                method: Method::Brightnessctl,
                max_brightness: None,
            });
        }

        // Try xbacklight
        if Self::has_command("xbacklight") {
            info!("Using xbacklight for brightness control");
            return Ok(Self {
                method: Method::Xbacklight,
                max_brightness: None,
            });
        }

        // Try sysfs
        if let Some((path, max)) = Self::find_sysfs_backlight()? {
            info!("Using sysfs at {:?} for brightness control", path);
            return Ok(Self {
                method: Method::Sysfs(path),
                max_brightness: Some(max),
            });
        }

        anyhow::bail!("No brightness control method available")
    }

    fn has_command(cmd: &str) -> bool {
        Command::new("which")
            .arg(cmd)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn find_sysfs_backlight() -> Result<Option<(PathBuf, i32)>> {
        let backlight_dir = PathBuf::from("/sys/class/backlight");
        if !backlight_dir.exists() {
            return Ok(None);
        }

        for entry in fs::read_dir(&backlight_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if we can read max_brightness
            let max_path = path.join("max_brightness");
            let brightness_path = path.join("brightness");

            if max_path.exists() && brightness_path.exists() {
                let max: i32 = fs::read_to_string(&max_path)?
                    .trim()
                    .parse()
                    .context("Failed to parse max_brightness")?;

                return Ok(Some((path, max)));
            }
        }

        Ok(None)
    }

    /// Get current brightness (0-100)
    pub fn get(&self) -> Result<i32> {
        match &self.method {
            Method::Brightnessctl => {
                let output = Command::new("brightnessctl")
                    .arg("get")
                    .output()
                    .context("Failed to run brightnessctl")?;

                let current: i32 = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .context("Failed to parse brightness")?;

                let max_output = Command::new("brightnessctl")
                    .arg("max")
                    .output()
                    .context("Failed to get max brightness")?;

                let max: i32 = String::from_utf8_lossy(&max_output.stdout)
                    .trim()
                    .parse()
                    .context("Failed to parse max brightness")?;

                Ok((current * 100) / max)
            }

            Method::Xbacklight => {
                let output = Command::new("xbacklight")
                    .arg("-get")
                    .output()
                    .context("Failed to run xbacklight")?;

                let brightness: f32 = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .context("Failed to parse brightness")?;

                Ok(brightness as i32)
            }

            Method::Sysfs(path) => {
                let current: i32 = fs::read_to_string(path.join("brightness"))?
                    .trim()
                    .parse()
                    .context("Failed to parse brightness")?;

                let max = self.max_brightness.unwrap_or(255);
                Ok((current * 100) / max)
            }
        }
    }

    /// Set brightness (0-100)
    pub fn set(&self, percent: i32) -> Result<()> {
        let percent = percent.clamp(1, 100);
        debug!("Setting brightness to {}%", percent);

        match &self.method {
            Method::Brightnessctl => {
                let status = Command::new("brightnessctl")
                    .arg("set")
                    .arg(format!("{}%", percent))
                    .status()
                    .context("Failed to run brightnessctl")?;

                if !status.success() {
                    // Try with sudo
                    let status = Command::new("sudo")
                        .arg("-n")
                        .arg("brightnessctl")
                        .arg("set")
                        .arg(format!("{}%", percent))
                        .status()?;

                    if !status.success() {
                        warn!("Failed to set brightness");
                    }
                }
            }

            Method::Xbacklight => {
                Command::new("xbacklight")
                    .arg("-set")
                    .arg(percent.to_string())
                    .status()
                    .context("Failed to run xbacklight")?;
            }

            Method::Sysfs(path) => {
                let max = self.max_brightness.unwrap_or(255);
                let value = (percent * max) / 100;

                let brightness_path = path.join("brightness");

                // Try direct write
                if fs::write(&brightness_path, value.to_string()).is_err() {
                    // Fall back to sudo
                    Command::new("sudo")
                        .arg("-n")
                        .arg("tee")
                        .arg(&brightness_path)
                        .stdin(std::process::Stdio::piped())
                        .output()
                        .context("Failed to write brightness")?;
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
        // ls should exist on any Linux system
        assert!(BrightnessControl::has_command("ls"));
        assert!(!BrightnessControl::has_command("nonexistent_command_xyz"));
    }
}
