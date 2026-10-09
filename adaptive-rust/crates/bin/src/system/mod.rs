//! System control for brightness and volume
//!
//! Provides Linux-specific controls using brightnessctl and amixer.

mod brightness;
mod volume;

pub use brightness::BrightnessControl;
pub(crate) use brightness::BrightnessReadback;
pub use volume::VolumeControl;

#[cfg(target_os = "linux")]
pub(crate) mod command_linux;

#[cfg(any(target_os = "windows", test))]
pub(crate) fn checked_percent(success: bool, text: &str, label: &str) -> anyhow::Result<i32> {
    anyhow::ensure!(success, "{label} command failed");
    let value: f64 = text.trim().parse()?;
    anyhow::ensure!(
        value.is_finite() && (0.0..=100.0).contains(&value),
        "Invalid {label} readback"
    );
    Ok(value.round() as i32)
}
#[cfg(test)]
mod command_result_tests {
    #[test]
    fn windows_command_failure_and_invalid_percent_are_errors_not_fifty() {
        use super::checked_percent;
        assert!(checked_percent(false, "50", "volume").is_err());
        for text in ["", "NaN", "inf", "-1", "101", "50\n60", "unavailable"] {
            assert!(checked_percent(true, text, "volume").is_err());
            assert!(checked_percent(true, text, "brightness").is_err());
        }
        assert_eq!(checked_percent(true, "50", "volume").unwrap(), 50);
        assert_eq!(checked_percent(true, "0", "volume").unwrap(), 0);
    }
}
