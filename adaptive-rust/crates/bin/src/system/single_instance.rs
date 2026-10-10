//! One controller per logon session on Windows. The sunrise/sunset tasks and the
//! 10-minute task can start together; a later run exits instead of fighting the
//! first over brightness and volume. The OS closes the handle if a run is killed.
use anyhow::{Context, Result};
use std::ffi::c_void;

const INSTANCE_NAME: &str = r"Local\AdaptiveBrightnessController";
const ERROR_ALREADY_EXISTS: i32 = 183;

#[link(name = "kernel32")]
extern "system" {
    fn CreateMutexW(attributes: *const c_void, initial_owner: i32, name: *const u16)
        -> *mut c_void;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

pub struct InstanceGuard(*mut c_void);

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful CreateMutexW and is closed once.
        unsafe { CloseHandle(self.0) };
    }
}

/// None when another controller in this session already runs.
pub fn acquire() -> Result<Option<InstanceGuard>> {
    acquire_named(INSTANCE_NAME)
}

fn acquire_named(name: &str) -> Result<Option<InstanceGuard>> {
    let name: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
    // SAFETY: `name` is a live NUL-terminated UTF-16 string; null attributes and
    // no initial ownership only open or create the named object.
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    // Read the creation status before any other call can overwrite it.
    let status = std::io::Error::last_os_error();
    if handle.is_null() {
        return Err(status).context("Creating controller instance mutex");
    }
    let guard = InstanceGuard(handle);
    Ok((status.raw_os_error() != Some(ERROR_ALREADY_EXISTS)).then_some(guard))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn second_run_in_session_is_refused_until_first_exits() {
        let name = format!(
            r"Local\AdaptiveBrightnessController-test-{}",
            std::process::id()
        );
        let first = acquire_named(&name)
            .unwrap()
            .expect("first run owns the name");
        assert!(acquire_named(&name).unwrap().is_none());
        drop(first);
        assert!(acquire_named(&name).unwrap().is_some());
    }
}
