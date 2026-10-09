//! Linux volume control restored from 5094193; bounded commands and checked readback.
use super::super::command_linux;
use anyhow::{Context, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Amixer,
    Pactl,
    Wpctl,
}
pub struct VolumeControl {
    method: Method,
}

impl VolumeControl {
    pub fn new() -> Result<Self> {
        Ok(Self {
            method: select_method(|method| Self { method }.get())?,
        })
    }
    pub fn get(&self) -> Result<i32> {
        let (program, args): (&str, &[&str]) = match self.method {
            Method::Amixer => ("/usr/bin/amixer", &["get", "Master"]),
            Method::Pactl => ("/usr/bin/pactl", &["get-sink-volume", "@DEFAULT_SINK@"]),
            Method::Wpctl => ("/usr/bin/wpctl", &["get-volume", "@DEFAULT_AUDIO_SINK@"]),
        };
        parse_volume(self.method, &command_linux::output(program, args)?)
    }
    pub fn set(&self, percent: i32) -> Result<()> {
        let percent = percent.clamp(0, 100);
        let value = match self.method {
            Method::Wpctl => (percent as f64 / 100.0).to_string(),
            _ => format!("{percent}%"),
        };
        let (program, args): (&str, Vec<&str>) = match self.method {
            Method::Amixer => ("/usr/bin/amixer", vec!["set", "Master", &value]),
            Method::Pactl => (
                "/usr/bin/pactl",
                vec!["set-sink-volume", "@DEFAULT_SINK@", &value],
            ),
            Method::Wpctl => (
                "/usr/bin/wpctl",
                vec!["set-volume", "@DEFAULT_AUDIO_SINK@", &value],
            ),
        };
        command_linux::output(program, &args)?;
        anyhow::ensure!(
            (self.get()? - percent).abs() <= 1,
            "Volume readback mismatch"
        );
        Ok(())
    }
}

fn select_method<F>(mut read: F) -> Result<Method>
where
    F: FnMut(Method) -> Result<i32>,
{
    for method in [Method::Pactl, Method::Wpctl, Method::Amixer] {
        if read(method).is_ok() {
            return Ok(method);
        }
    }
    anyhow::bail!("No readable Linux volume control available")
}

fn parse_volume(method: Method, text: &str) -> Result<i32> {
    let value = match method {
        Method::Amixer => text
            .split('[')
            .skip(1)
            .find_map(|part| {
                part.split_once("%]")
                    .and_then(|(s, _)| s.parse::<f64>().ok())
            })
            .context("Invalid amixer volume")?,
        Method::Pactl => text
            .split_whitespace()
            .find_map(|part| part.strip_suffix('%').and_then(|s| s.parse::<f64>().ok()))
            .context("Invalid pactl volume")?,
        Method::Wpctl => {
            text.strip_prefix("Volume: ")
                .and_then(|s| s.split_whitespace().next())
                .context("Invalid wpctl volume")?
                .parse::<f64>()?
                * 100.0
        }
    };
    anyhow::ensure!(
        value.is_finite() && (0.0..=100.0).contains(&value),
        "Invalid volume range"
    );
    Ok(value.round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn active_session_volume_is_preferred_to_readable_alsa() {
        let mut calls = Vec::new();
        assert_eq!(
            select_method(|m| {
                calls.push(m);
                Ok(50)
            })
            .unwrap(),
            Method::Pactl
        );
        assert_eq!(calls, [Method::Pactl]);
        calls.clear();
        assert_eq!(
            select_method(|m| {
                calls.push(m);
                if m == Method::Pactl {
                    anyhow::bail!("no pulse")
                } else {
                    Ok(50)
                }
            })
            .unwrap(),
            Method::Wpctl
        );
        assert_eq!(calls, [Method::Pactl, Method::Wpctl]);
        calls.clear();
        assert_eq!(
            select_method(|m| {
                calls.push(m);
                if m == Method::Amixer {
                    Ok(50)
                } else {
                    anyhow::bail!("no session")
                }
            })
            .unwrap(),
            Method::Amixer
        );
        assert_eq!(calls, [Method::Pactl, Method::Wpctl, Method::Amixer]);
        assert!(select_method(|_| anyhow::bail!("no control")).is_err());
    }
    #[test]
    fn actual_linux_formats_and_zero_parse() {
        assert_eq!(
            parse_volume(Method::Amixer, "Mono: Playback 0 [0%] [-inf dB] [on]").unwrap(),
            0
        );
        assert_eq!(
            parse_volume(Method::Pactl, "Volume: front-left: 32768 / 50% / -6 dB").unwrap(),
            50
        );
        assert_eq!(
            parse_volume(Method::Wpctl, "Volume: 0.35 [MUTED]").unwrap(),
            35
        );
    }
    #[test]
    fn malformed_nonfinite_and_out_of_range_are_not_confirmed() {
        for input in [
            "",
            "Volume: NaN",
            "Volume: inf",
            "Volume: -0.1",
            "Volume: 1.5",
        ] {
            assert!(parse_volume(Method::Wpctl, input).is_err());
        }
        assert!(parse_volume(Method::Amixer, "[on]").is_err());
        assert!(parse_volume(Method::Pactl, "Volume: unknown").is_err());
    }
}
