#[cfg(target_os = "linux")]
#[path = "camera_linux.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "camera_windows.rs"]
mod platform;
pub use platform::Camera;
