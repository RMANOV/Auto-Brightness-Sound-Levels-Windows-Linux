//! Standalone Adaptive Brightness/Volume Controller
//!
//! High-performance native implementation with lock-free channels.

mod camera;
mod audio_capture;
mod controller;
mod system;

use anyhow::Result;
use nix::sys::signal::{self, Signal, SigHandler};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Global shutdown flag for signal handling
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Signal handler for graceful shutdown
extern "C" fn handle_signal(_: i32) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();

    info!("Adaptive Brightness/Volume Controller v{}", env!("CARGO_PKG_VERSION"));
    info!("Starting up...");

    // Setup signal handlers
    unsafe {
        signal::signal(Signal::SIGTERM, SigHandler::Handler(handle_signal))?;
        signal::signal(Signal::SIGINT, SigHandler::Handler(handle_signal))?;
    }

    // Create and run controller
    let continuous = std::env::args().any(|a| a == "--continuous");
    let mut config = controller::ControllerConfig::default();
    config.auto_exit = !continuous;
    let mut controller = controller::Controller::new(config)?;

    info!("Controller initialized, entering main loop");
    if continuous {
        info!("Mode: continuous (Ctrl+C to stop)");
    }

    // Main loop
    while !SHUTDOWN.load(Ordering::SeqCst) {
        match controller.tick() {
            Ok(true) => break,  // Converged
            Ok(false) => {}
            Err(e) => tracing::error!("Controller error: {}", e),
        }
    }

    info!("Cleaning up...");
    controller.cleanup();
    info!("Cleanup complete, exiting");

    Ok(())
}
