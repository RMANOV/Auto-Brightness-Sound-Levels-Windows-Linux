//! Brightness control
//!
//! Windows: WMI via PowerShell (laptop backlight)
//! Falls back gracefully on desktop monitors.

use super::{optional_readback, BrightnessReadback};
use crate::system::checked_percent;
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
                let parsed = checked_percent(true, &stdout, "brightness").is_ok();
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

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub(crate) fn get_readback(&self) -> Result<Option<BrightnessReadback>> {
        optional_readback(self.available, || {
            Ok(BrightnessReadback {
                percent: self.get()? as f64,
                step_percent: 1.0,
            })
        })
    }

    pub fn get(&self) -> Result<i32> {
        anyhow::ensure!(self.available, "Windows brightness control unavailable");

        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness",
            ])
            .output()
            .context("Failed to query brightness")?;

        checked_percent(
            output.status.success(),
            &String::from_utf8_lossy(&output.stdout),
            "brightness",
        )
    }

    pub fn set(&self, percent: i32) -> Result<()> {
        anyhow::ensure!(self.available, "Windows brightness control unavailable");

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
            anyhow::bail!("Brightness set failed: {}", stderr.trim());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_wmi_is_capability_none_never_fifty_or_noop_ack() {
        let control = BrightnessControl { available: false };
        assert!(control.get_readback().unwrap().is_none());
        assert!(control.get().is_err());
        assert!(control.set(30).is_err());
    }
}
