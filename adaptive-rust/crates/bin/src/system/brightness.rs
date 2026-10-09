#[cfg(target_os = "linux")]
#[path = "brightness_linux.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "brightness_windows.rs"]
mod platform;
pub use platform::BrightnessControl;

/// Measured same-device percent and one raw hardware step, not desired state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BrightnessReadback {
    pub percent: f64,
    pub step_percent: f64,
}
impl BrightnessReadback {
    pub fn matches_target(self, target: f32) -> bool {
        target.is_finite()
            && self.percent.is_finite()
            && (0.0..=100.0).contains(&self.percent)
            && self.step_percent.is_finite()
            && (0.0..=100.0).contains(&self.step_percent)
            && self.step_percent > 0.0
            && (self.percent - (target.round() as f64).clamp(1.0, 100.0)).abs()
                <= self.step_percent + 1e-9
    }
}

pub(crate) fn optional_readback<F>(
    available: bool,
    mut read: F,
) -> anyhow::Result<Option<BrightnessReadback>>
where
    F: FnMut() -> anyhow::Result<BrightnessReadback>,
{
    if available {
        Ok(Some(read()?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    #[test]
    fn unavailable_brightness_never_queries_or_fabricates_a_readback() {
        let mut reads = 0;
        let result = optional_readback(false, || {
            reads += 1;
            Ok(BrightnessReadback {
                percent: 50.0,
                step_percent: 1.0,
            })
        })
        .unwrap();
        assert!(result.is_none());
        assert_eq!(reads, 0);
    }
    #[test]
    fn available_read_failure_is_an_error_not_capability_loss() {
        assert!(optional_readback(true, || anyhow::bail!("query failed")).is_err());
    }
    #[test]
    fn reachable_raw_step_matches_rounded_request_not_fractional_target() {
        let low_res = BrightnessReadback {
            percent: 100.0 / 3.0,
            step_percent: 100.0 / 9.0,
        };
        assert!(low_res.matches_target(30.4));
        assert!(BrightnessReadback {
            percent: 30.0,
            step_percent: 1.0
        }
        .matches_target(30.49));
        assert!(!BrightnessReadback {
            percent: 32.0,
            step_percent: 1.0
        }
        .matches_target(30.49));
        assert!(!BrightnessReadback {
            percent: 500.0 / 9.0,
            step_percent: 100.0 / 9.0
        }
        .matches_target(30.4));
        for target in [f32::NAN, f32::INFINITY] {
            assert!(!low_res.matches_target(target));
        }
        for step in [0.0, f64::NAN, f64::INFINITY, 101.0] {
            assert!(!BrightnessReadback {
                percent: 30.0,
                step_percent: step
            }
            .matches_target(30.0));
        }
    }
}
