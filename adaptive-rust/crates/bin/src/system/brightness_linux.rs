//! Linux backlight only: bind one sysfs backlight, never an implicit LED device.
use super::super::command_linux;
use anyhow::{Context, Result};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

pub struct BrightnessControl {
    device: String,
}

impl BrightnessControl {
    pub fn new() -> Result<Self> {
        let preferred = std::env::var("ADAPTIVE_BACKLIGHT_DEVICE").ok();
        let device = select_device(Path::new("/sys/class/backlight"), preferred.as_deref())?;
        let control = Self { device };
        control.get()?;
        Ok(control)
    }
    pub fn get(&self) -> Result<f64> {
        let (raw, max) = read_raw(&self.device, &mut command_linux::output)?;
        Ok(raw as f64 * 100.0 / max as f64)
    }
    pub fn set(&self, percent: i32) -> Result<()> {
        set_verified(&self.device, percent, &mut command_linux::output)
    }
}

fn select_device(root: &Path, preferred: Option<&str>) -> Result<String> {
    let valid = |name: &str| {
        !name.is_empty()
            && name != "."
            && name != ".."
            && !name.contains('/')
            && !name.contains('\\')
    };
    let mut devices = Vec::new();
    for entry in std::fs::read_dir(root).context("No sysfs backlight directory")? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("Non-UTF8 backlight name"))?;
        if valid(&name)
            && entry.path().join("brightness").exists()
            && entry.path().join("max_brightness").exists()
        {
            devices.push(name);
        }
    }
    devices.sort();
    if let Some(name) = preferred {
        anyhow::ensure!(
            valid(name) && devices.iter().any(|d| d == name),
            "Requested backlight is unavailable"
        );
        return Ok(name.to_owned());
    }
    anyhow::ensure!(
        devices.len() == 1,
        "Expected one backlight; set ADAPTIVE_BACKLIGHT_DEVICE explicitly (found {})",
        devices.len()
    );
    Ok(devices.remove(0))
}

fn read_raw<F>(device: &str, run: &mut F) -> Result<(u64, u64)>
where
    F: FnMut(&str, &[&str]) -> Result<String>,
{
    let raw: u64 = run(
        "/usr/bin/brightnessctl",
        &["--class=backlight", "--device", device, "get"],
    )?
    .trim()
    .parse()
    .context("Invalid brightness")?;
    let max: u64 = run(
        "/usr/bin/brightnessctl",
        &["--class=backlight", "--device", device, "max"],
    )?
    .trim()
    .parse()
    .context("Invalid maximum brightness")?;
    anyhow::ensure!(max > 0 && raw <= max, "Invalid backlight range");
    Ok((raw, max))
}

fn set_verified<F>(device: &str, percent: i32, run: &mut F) -> Result<()>
where
    F: FnMut(&str, &[&str]) -> Result<String>,
{
    let percent = percent.clamp(1, 100);
    let requested = format!("{percent}%");
    let args = ["--class=backlight", "--device", device, "set", &requested];
    if run("/usr/bin/brightnessctl", &args).is_err() {
        let mut sudo_args = vec!["-n", "/usr/bin/brightnessctl"];
        sudo_args.extend_from_slice(&args);
        run("/usr/bin/sudo", &sudo_args).context("Both brightness commands failed")?;
    }
    let (raw, max) = read_raw(device, run)?;
    // brightnessctl rounds percentages to raw integer hardware steps.
    // Permit at most one raw step, not a fabricated/requested software state.
    let actual = raw as f64 * 100.0 / max as f64;
    let tolerance = 100.0 / max as f64;
    anyhow::ensure!(
        (actual - percent as f64).abs() <= tolerance + 1e-9,
        "Backlight readback mismatch: requested {percent}%, actual {actual}%"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    fn script(
        mut replies: VecDeque<Result<String>>,
    ) -> impl FnMut(&str, &[&str]) -> Result<String> {
        move |program, args| {
            assert!(program == "/usr/bin/brightnessctl" || program == "/usr/bin/sudo");
            assert!(args.windows(2).any(|x| x == ["--device", "panel"]));
            assert!(args.contains(&"--class=backlight"));
            if program == "/usr/bin/sudo" {
                assert_eq!(args[0], "-n");
            }
            replies.pop_front().expect("unexpected command")
        }
    }
    fn ok(s: &str) -> Result<String> {
        Ok(s.to_owned())
    }
    #[test]
    fn both_failed_commands_cannot_report_applied_set() {
        let mut run = script(VecDeque::from([
            Err(anyhow::anyhow!("refused")),
            Err(anyhow::anyhow!("refused")),
        ]));
        assert!(set_verified("panel", 30, &mut run).is_err());
    }
    #[test]
    fn unavailable_actuator_cannot_report_confirmed_readback() {
        let mut run = script(VecDeque::from([Err(anyhow::anyhow!("absent"))]));
        assert!(read_raw("panel", &mut run).is_err());
    }
    #[test]
    fn successful_write_requires_matching_same_device_readback() {
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([ok(""), ok("30"), ok("100")]))
        )
        .is_ok());
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([ok(""), ok("80"), ok("100")]))
        )
        .is_err());
    }
    #[test]
    fn sudo_success_still_requires_readback() {
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([
                Err(anyhow::anyhow!("permission")),
                ok(""),
                ok("30"),
                ok("100")
            ]))
        )
        .is_ok());
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([
                Err(anyhow::anyhow!("permission")),
                ok(""),
                Err(anyhow::anyhow!("readback"))
            ]))
        )
        .is_err());
    }
    #[test]
    fn malformed_zero_max_and_out_of_range_are_errors() {
        for (raw, max) in [
            ("nan", "100"),
            ("30", "0"),
            ("101", "100"),
            ("-1", "100"),
            ("30", "inf"),
        ] {
            assert!(read_raw("panel", &mut script(VecDeque::from([ok(raw), ok(max)]))).is_err());
        }
        assert_eq!(
            read_raw("panel", &mut script(VecDeque::from([ok("0"), ok("100")]))).unwrap(),
            (0, 100)
        );
    }
    #[test]
    fn quantization_tolerance_is_one_raw_step() {
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([ok(""), ok("3"), ok("9")]))
        )
        .is_ok());
        assert!(set_verified(
            "panel",
            30,
            &mut script(VecDeque::from([ok(""), ok("5"), ok("9")]))
        )
        .is_err());
    }
    #[test]
    fn backlight_selection_rejects_led_or_ambiguous_default() {
        let root: PathBuf =
            std::env::temp_dir().join(format!("adaptive-backlight-test-{}", std::process::id()));
        std::fs::create_dir_all(root.join("panel")).unwrap();
        for f in ["brightness", "max_brightness"] {
            std::fs::write(root.join("panel").join(f), "100").unwrap();
        }
        assert_eq!(select_device(&root, None).unwrap(), "panel");
        assert!(select_device(&root, Some("../led")).is_err());
        assert!(select_device(&root, Some("led")).is_err());
        std::fs::create_dir_all(root.join("second")).unwrap();
        for f in ["brightness", "max_brightness"] {
            std::fs::write(root.join("second").join(f), "100").unwrap();
        }
        assert!(select_device(&root, None).is_err());
        assert_eq!(select_device(&root, Some("panel")).unwrap(), "panel");
        std::fs::remove_dir_all(root).unwrap();
    }
}
