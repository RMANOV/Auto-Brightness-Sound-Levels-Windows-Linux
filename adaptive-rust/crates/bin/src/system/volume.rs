#[cfg(target_os = "linux")]
#[path = "volume_linux.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "volume_windows.rs"]
mod platform;
pub use platform::VolumeControl;
