//! Noninteractive bounded commands. No shell expansion or privilege policy changes.
use anyhow::{Context, Result};
use std::process::Command;

pub fn output(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new("/usr/bin/timeout")
        .args(["--kill-after=1s", "3s", program])
        .args(args)
        .output()
        .with_context(|| format!("Failed to run {program}"))?;
    anyhow::ensure!(
        output.status.success(),
        "{program} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout).context("Command output is not UTF-8")
}
