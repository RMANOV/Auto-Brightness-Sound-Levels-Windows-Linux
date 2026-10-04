//! Standalone Adaptive Brightness/Volume Controller
//!
//! Cross-platform implementation with lock-free channels.
//! Windows: sun-position ambient light + WMI brightness + Core Audio volume
//! Linux: v4l2 camera + brightnessctl + amixer

mod audio_capture;
mod camera;
mod controller;
mod system;

use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    // Initialize logging
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();

    info!(
        "Adaptive Brightness/Volume Controller v{}",
        env!("CARGO_PKG_VERSION")
    );
    info!("Platform: {}", std::env::consts::OS);
    info!("Starting up...");

    // Setup cross-platform signal handler
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::SeqCst);
    })?;

    // Create and run controller
    let continuous = std::env::args().any(|a| a == "--continuous");
    let mut config = controller::ControllerConfig::default();
    config.auto_exit = !continuous;
    let mut ctrl = controller::Controller::new(config)?;

    info!("Controller initialized, entering main loop");
    if continuous {
        info!("Mode: continuous (Ctrl+C to stop)");
    }

    // Main loop
    while !shutdown.load(Ordering::SeqCst) {
        match ctrl.tick() {
            Ok(true) => break, // Converged
            Ok(false) => {}
            Err(e) => tracing::error!("Controller error: {}", e),
        }
    }

    info!("Cleaning up...");
    ctrl.cleanup();
    info!("Cleanup complete, exiting");

    Ok(())
}
