#[cfg(target_os = "linux")]
#[path = "brightness_linux.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "brightness_windows.rs"]
mod platform;
pub use platform::BrightnessControl;
