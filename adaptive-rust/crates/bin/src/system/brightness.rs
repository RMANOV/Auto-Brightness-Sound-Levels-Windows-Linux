//! Brightness control
//!
//! Windows: WMI via PowerShell (laptop backlight)
//! Falls back gracefully on desktop monitors.

use anyhow::{Context, Result};
use std::process::Command;
use tracing::{debug, info, warn};

pub struct BrightnessControl {
    available: bool,
}

impl BrightnessControl {
    pub fn new() -> Result<Self> {
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness",
            ])
            .output();

        let available = match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let parsed = stdout.trim().parse::<i32>().is_ok();
                if parsed {
                    info!("WMI brightness control available (laptop backlight)");
                }
                parsed
            }
            _ => false,
        };

        if !available {
            warn!("WMI brightness not available — brightness adjustment disabled");
            warn!("(This is normal on desktop monitors without DDC/CI support)");
        }

        Ok(Self { available })
    }

    pub fn get(&self) -> Result<i32> {
        if !self.available {
            return Ok(50);
        }

        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness",
            ])
            .output()
            .context("Failed to query brightness")?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(val) = stdout.trim().parse::<i32>() {
                return Ok(val);
            }
        }

        Ok(50)
    }

    pub fn set(&self, percent: i32) -> Result<()> {
        if !self.available {
            debug!("Brightness set skipped (not available)");
            return Ok(());
        }

        let percent = percent.clamp(1, 100);
        debug!("Setting brightness to {}%", percent);

        let cmd = format!(
            "Invoke-CimMethod -InputObject (Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods) -MethodName WmiSetBrightness -Arguments @{{Timeout=1; Brightness={}}}",
            percent
        );

        let output = Command::new("powershell")
            .args(["-NoProfile", "-Command", &cmd])
            .output()
            .context("Failed to set brightness")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Brightness set failed: {}", stderr.trim());
        }

        Ok(())
    }
}
