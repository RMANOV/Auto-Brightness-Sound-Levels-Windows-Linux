//! Standalone Adaptive Brightness/Volume Controller
//!
//! Cross-platform implementation with lock-free channels.
//! Windows: sun-position ambient light + WMI brightness + Core Audio volume
//! Linux: v4l2 camera + brightnessctl + amixer

mod audio_capture;
mod camera;
mod controller;
mod system;

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
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

    // Sunrise/sunset and 10-minute tasks may overlap; only one run adjusts.
    #[cfg(target_os = "windows")]
    let _instance = match system::single_instance::acquire()? {
        Some(guard) => guard,
        None => {
            info!("Another controller run is active in this session; skipping");
            return Ok(());
        }
    };

    // Task Scheduler's time limit stops the conhost launcher, not this process,
    // so a stuck scheduled run ends itself and releases the instance guard.
    let args: Vec<String> = std::env::args().collect();
    if let Some(limit) = max_runtime(&args)? {
        std::thread::spawn(move || {
            std::thread::sleep(limit);
            tracing::error!("Run limit of {}s reached; exiting", limit.as_secs());
            std::process::exit(1);
        });
    }

    // Setup cross-platform signal handler
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::SeqCst);
    })?;

    // Create and run controller
    let continuous = args.iter().any(|a| a == "--continuous");
    let mut config = controller::ControllerConfig::default();
    config.auto_exit = !continuous;
    let mut ctrl = controller::Controller::new(config)?;

    info!("Controller initialized, entering main loop");
    if continuous {
        info!("Mode: continuous (Ctrl+C to stop)");
    }

    // Main loop
    let mut converged = false;
    let mut failure = None;
    while !shutdown.load(Ordering::SeqCst) {
        match ctrl.tick() {
            Ok(true) => {
                converged = true;
                break;
            } // Confirmed convergence
            Ok(false) => {}
            Err(e) => {
                tracing::error!("Controller error: {}", e);
                failure = Some(e);
                break;
            }
        }
    }

    info!("Cleaning up...");
    ctrl.cleanup();
    info!("Cleanup complete, exiting");

    if let Some(error) = failure {
        return Err(error);
    }
    anyhow::ensure!(
        continuous || converged,
        "Controller stopped without confirmed convergence"
    );
    Ok(())
}

/// `--max-runtime-secs N` caps one scheduled run; without it a run is uncapped.
fn max_runtime(args: &[String]) -> Result<Option<Duration>> {
    let Some(flag) = args.iter().position(|a| a == "--max-runtime-secs") else {
        return Ok(None);
    };
    let secs: u64 = args
        .get(flag + 1)
        .and_then(|value| value.parse().ok())
        .context("--max-runtime-secs needs a whole number of seconds")?;
    anyhow::ensure!(secs > 0, "--max-runtime-secs must be positive");
    Ok(Some(Duration::from_secs(secs)))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| arg.to_string()).collect()
    }
    #[test]
    fn run_limit_is_optional_and_must_be_positive_seconds() {
        assert_eq!(max_runtime(&args(&["exe"])).unwrap(), None);
        assert_eq!(
            max_runtime(&args(&["exe", "--max-runtime-secs", "480"])).unwrap(),
            Some(Duration::from_secs(480))
        );
        for bad in [
            &["exe", "--max-runtime-secs"][..],
            &["exe", "--max-runtime-secs", "0"],
            &["exe", "--max-runtime-secs", "8m"],
        ] {
            assert!(max_runtime(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
