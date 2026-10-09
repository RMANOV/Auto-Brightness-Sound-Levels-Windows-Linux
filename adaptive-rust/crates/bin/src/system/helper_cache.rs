//! Portable cache lifecycle; Windows supplies compiler, non-device self-test and publisher.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_BUILD: AtomicU64 = AtomicU64::new(0);

pub(crate) struct BuildDirectory(PathBuf);
impl BuildDirectory {
    pub(crate) fn new(parent: &Path) -> Result<Self> {
        for _ in 0..64 {
            let path = parent.join(format!(
                ".volume-helper-build-{}-{}",
                std::process::id(),
                NEXT_BUILD.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).context("Creating private helper build directory"),
            }
        }
        anyhow::bail!("Could not create an exclusive helper build directory")
    }
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for BuildDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn ensure_helper<F, V, P>(
    path: &Path,
    mut compile: F,
    mut validate: V,
    mut publish: P,
) -> Result<()>
where
    F: FnMut(&Path) -> Result<()>,
    V: FnMut(&Path) -> Result<()>,
    P: FnMut(&Path, &Path) -> Result<()>,
{
    if path.exists() && validate(path).is_ok() {
        return Ok(());
    }
    let build = BuildDirectory::new(path.parent().context("Helper cache has no parent")?)?;
    let candidate = build.path().join("volume_helper.exe");
    compile(&candidate)?;
    validate(&candidate).context("Compiled helper failed non-device self-test")?;
    // Same-volume replacement: the incumbent is never removed first, even on error.
    publish(&candidate, path).context("Publishing verified helper")?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn atomic_publish(from: &Path, to: &Path) -> Result<()> {
    std::fs::rename(from, to)?;
    Ok(())
}

#[cfg(target_os = "windows")]
pub(crate) fn atomic_publish(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "Kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    let encode = |path: &Path| -> Result<Vec<u16>> {
        let mut value: Vec<u16> = path.as_os_str().encode_wide().collect();
        anyhow::ensure!(!value.contains(&0), "Invalid helper filename");
        value.push(0);
        Ok(value)
    };
    let from = encode(from)?;
    let to = encode(to)?;
    // SAFETY: both arrays are live, NUL-terminated UTF-16 strings. The Win32
    // signature is LPCWSTR, LPCWSTR, DWORD -> BOOL. No copy-across-volume flag.
    let success = unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0x1 | 0x8) };
    if success == 0 {
        return Err(std::io::Error::last_os_error()).context("Atomic helper replacement");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "adaptive-cache-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn cache(&self) -> std::path::PathBuf {
            self.0.join("helper.exe")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn validate(p: &Path) -> Result<()> {
        anyhow::ensure!(std::fs::read(p)? == b"valid", "invalid cache");
        Ok(())
    }
    fn publish(from: &Path, to: &Path) -> Result<()> {
        atomic_publish(from, to)
    }
    #[test]
    fn corrupt_cache_is_rebuilt_and_checked_before_replacement() {
        let t = Temp::new();
        let cache = t.cache();
        std::fs::write(&cache, b"broken").unwrap();
        let mut compilations = 0;
        ensure_helper(
            &cache,
            |candidate| {
                compilations += 1;
                assert_ne!(candidate, cache);
                assert_eq!(std::fs::read(&cache)?, b"broken");
                std::fs::write(candidate, b"valid")?;
                Ok(())
            },
            validate,
            publish,
        )
        .unwrap();
        assert_eq!(compilations, 1);
        assert_eq!(std::fs::read(&cache).unwrap(), b"valid");
        assert_eq!(std::fs::read_dir(&t.0).unwrap().count(), 1);
    }
    #[test]
    fn interrupted_compilation_never_publishes_partial_executable() {
        let t = Temp::new();
        let cache = t.cache();
        assert!(ensure_helper(
            &cache,
            |candidate| {
                std::fs::write(candidate, b"partial")?;
                anyhow::bail!("compiler interrupted")
            },
            validate,
            publish
        )
        .is_err());
        assert!(!cache.exists());
        assert_eq!(std::fs::read_dir(&t.0).unwrap().count(), 0);
    }
    #[test]
    fn compile_validation_and_publish_failures_preserve_incumbent() {
        for failure in ["compile", "validate", "publish"] {
            let t = Temp::new();
            let cache = t.cache();
            std::fs::write(&cache, b"broken").unwrap();
            let result = ensure_helper(
                &cache,
                |candidate| {
                    std::fs::write(
                        candidate,
                        if failure == "validate" {
                            b"wrong".as_slice()
                        } else {
                            b"valid".as_slice()
                        },
                    )?;
                    if failure == "compile" {
                        anyhow::bail!("interrupted compilation");
                    }
                    Ok(())
                },
                validate,
                |from, to| {
                    if failure == "publish" {
                        anyhow::bail!("publication denied");
                    }
                    atomic_publish(from, to)
                },
            );
            assert!(result.is_err(), "{failure}");
            assert_eq!(std::fs::read(&cache).unwrap(), b"broken");
            assert_eq!(std::fs::read_dir(&t.0).unwrap().count(), 1);
        }
    }
    #[test]
    fn staging_directories_are_exclusive_and_cleanup_only_owned_paths() {
        let t = Temp::new();
        let sentinel = t.0.join("unrelated");
        std::fs::write(&sentinel, b"keep").unwrap();
        let a = BuildDirectory::new(&t.0).unwrap();
        let b = BuildDirectory::new(&t.0).unwrap();
        assert_ne!(a.path(), b.path());
        assert!(a.path().is_dir());
        assert!(b.path().is_dir());
        drop(a);
        assert!(b.path().is_dir());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"keep");
        drop(b);
        assert_eq!(std::fs::read_dir(&t.0).unwrap().count(), 1);
    }
    #[test]
    fn valid_cache_is_reused_without_compiler_or_publisher() {
        let t = Temp::new();
        let cache = t.cache();
        std::fs::write(&cache, b"valid").unwrap();
        ensure_helper(
            &cache,
            |_| panic!("valid cache must not compile"),
            validate,
            |_, _| panic!("must not publish"),
        )
        .unwrap();
    }
}
