//! System control for brightness and volume
//!
//! Provides Linux-specific controls using brightnessctl and amixer.

mod brightness;
mod volume;

pub use brightness::BrightnessControl;
pub use volume::VolumeControl;

#[cfg(target_os = "linux")]
pub(crate) mod command_linux;
