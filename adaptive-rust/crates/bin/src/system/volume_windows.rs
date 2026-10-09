//! Volume control
//!
//! Windows: Core Audio API via a pre-compiled C# helper.
//! The helper is compiled once from embedded source using csc.exe,
//! then reused for fast (<50ms) volume get/set operations.

use crate::system::{
    checked_percent,
    helper_cache::{atomic_publish, ensure_helper},
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};

/// Embedded C# source for the volume helper executable.
/// COM interfaces must declare ALL methods in vtable order (after IUnknown's 3).
const VOLUME_HELPER_CS: &str = r#"
using System;
using System.Runtime.InteropServices;

// IMMDeviceEnumerator — vtable: IUnknown(3) + EnumAudioEndpoints, GetDefaultAudioEndpoint, ...
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int dwStateMask, out IntPtr ppDevices);
    [PreserveSig]
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppEndpoint);
    int GetDevice([MarshalAs(UnmanagedType.LPWStr)] string pwstrId, out IMMDevice ppDevice);
    int RegisterEndpointNotificationCallback(IntPtr pClient);
    int UnregisterEndpointNotificationCallback(IntPtr pClient);
}

// IMMDevice — vtable: IUnknown(3) + Activate, OpenPropertyStore, GetId, GetState
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    [PreserveSig]
    int Activate([MarshalAs(UnmanagedType.LPStruct)] Guid iid, int dwClsCtx,
                 IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
    int OpenPropertyStore(int stgmAccess, [MarshalAs(UnmanagedType.Interface)] out object ppProperties);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string ppstrId);
    int GetState(out int pdwState);
}

// IAudioEndpointVolume — vtable: IUnknown(3) + all methods in order
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
    int RegisterControlChangeNotify(IntPtr pNotify);
    int UnregisterControlChangeNotify(IntPtr pNotify);
    int GetChannelCount(out int pnChannelCount);
    int SetMasterVolumeLevel(float fLevelDB, Guid pguidEventContext);
    [PreserveSig]
    int SetMasterVolumeLevelScalar(float fLevel, IntPtr pguidEventContext);
    int GetMasterVolumeLevel(out float pfLevelDB);
    [PreserveSig]
    int GetMasterVolumeLevelScalar(out float pfLevel);
    int SetChannelVolumeLevel(int nChannel, float fLevelDB, Guid pguidEventContext);
    int SetChannelVolumeLevelScalar(int nChannel, float fLevel, Guid pguidEventContext);
    int GetChannelVolumeLevel(int nChannel, out float pfLevelDB);
    int GetChannelVolumeLevelScalar(int nChannel, out float pfLevel);
    int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, Guid pguidEventContext);
    int GetMute([MarshalAs(UnmanagedType.Bool)] out bool pbMute);
    int GetVolumeStepInfo(out int pnStep, out int pnStepCount);
    int VolumeStepUp(Guid pguidEventContext);
    int VolumeStepDown(Guid pguidEventContext);
    int QueryHardwareSupport(out int pdwHardwareSupportMask);
    int GetVolumeRange(out float pflVolumeMindB, out float pflVolumeMaxdB, out float pflVolumeIncrementdB);
}

[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MMDeviceEnumerator {}

class VolumeHelper {
    static IAudioEndpointVolume GetEndpointVolume() {
        var enumerator = (IMMDeviceEnumerator)(new MMDeviceEnumerator());
        IMMDevice device;
        Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(0 /* eRender */, 1 /* eMultimedia */, out device));
        Guid iid = typeof(IAudioEndpointVolume).GUID;
        object activated;
        Marshal.ThrowExceptionForHR(device.Activate(iid, 1 /* CLSCTX_ALL */, IntPtr.Zero, out activated));
        return (IAudioEndpointVolume)activated;
    }

    static int Main(string[] args) {
        if (args.Length < 1) {
            Console.Error.WriteLine("Usage: volume_helper.exe get | set <0-100>");
            return 1;
        }
        // This validates the cached executable independently of endpoint availability.
        if (args[0] == "self-test") {
            Console.WriteLine("adaptive-volume-checked-v3");
            return 0;
        }
        try {
            if (args[0] == "get") {
                float level;
                Marshal.ThrowExceptionForHR(GetEndpointVolume().GetMasterVolumeLevelScalar(out level));
                Console.WriteLine(Math.Round(level * 100));
            } else if (args[0] == "set" && args.Length >= 2) {
                float pct;
                if (!float.TryParse(args[1], out pct)) { Console.Error.WriteLine("Invalid number"); return 1; }
                Marshal.ThrowExceptionForHR(GetEndpointVolume().SetMasterVolumeLevelScalar(
                    Math.Max(0f, Math.Min(1f, pct / 100f)), IntPtr.Zero));
            } else {
                Console.Error.WriteLine("Unknown command");
                return 1;
            }
        } catch (Exception ex) {
            Console.Error.WriteLine("Error: " + ex.Message);
            return 2;
        }
        return 0;
    }
}
"#;

pub struct VolumeControl {
    helper_path: PathBuf,
}

impl VolumeControl {
    pub fn new() -> Result<Self> {
        let app_dir = get_app_data_dir()?;
        std::fs::create_dir_all(&app_dir)?;

        // The v3 self-test distinguishes corrupt code from unavailable audio.
        // A working v2 cache remains untouched; v3 is compiled beside the final
        // destination and published only after successful compilation/self-test.
        let exe_path = app_dir.join("volume_helper_checked_v3.exe");
        ensure_helper(&exe_path, compile_helper, validate_helper, atomic_publish)?;

        let control = Self {
            helper_path: exe_path,
        };
        let current = control.get()?;
        info!("Volume control ready (current: {}%)", current);
        Ok(control)
    }

    pub fn get(&self) -> Result<i32> {
        let output = Command::new(&self.helper_path)
            .arg("get")
            .output()
            .context("Failed to run volume helper")?;

        checked_percent(
            output.status.success(),
            &String::from_utf8_lossy(&output.stdout),
            "volume",
        )
    }

    pub fn set(&self, percent: i32) -> Result<()> {
        let percent = percent.clamp(0, 100);
        debug!("Setting volume to {}%", percent);

        let output = Command::new(&self.helper_path)
            .args(["set", &percent.to_string()])
            .output()
            .context("Failed to run volume helper")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Volume set failed: {}", stderr.trim());
        }

        anyhow::ensure!(
            (self.get()? - percent).abs() <= 1,
            "Volume readback mismatch"
        );
        Ok(())
    }
}

fn compile_helper(exe_path: &Path) -> Result<()> {
    info!("Compiling volume helper in private staging directory...");
    let cs_path = exe_path.with_extension("cs");
    std::fs::write(&cs_path, VOLUME_HELPER_CS)?;
    let csc = find_csc()?;
    let status = Command::new(&csc)
        .args([
            &format!("/out:{}", exe_path.display()),
            "/target:exe",
            "/optimize+",
            "/nologo",
            &cs_path.display().to_string(),
        ])
        .output()
        .with_context(|| format!("Failed to run csc.exe at {:?}", csc))?;
    anyhow::ensure!(
        status.status.success(),
        "C# compilation failed: stdout: {} stderr: {}",
        String::from_utf8_lossy(&status.stdout).trim(),
        String::from_utf8_lossy(&status.stderr).trim()
    );
    Ok(())
}
fn validate_helper(path: &Path) -> Result<()> {
    let output = Command::new(path)
        .arg("self-test")
        .output()
        .context("Helper self-test could not launch")?;
    anyhow::ensure!(
        output.status.success()
            && String::from_utf8_lossy(&output.stdout).trim() == "adaptive-volume-checked-v3",
        "Invalid volume helper self-test"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_csharp_helper_compiles_and_self_tests_without_audio() {
        // Windows CI proves the embedded C# compile + non-device command. No get/set,
        // endpoint enumeration, controller launch, or device operation is performed.
        let temp = crate::system::helper_cache::BuildDirectory::new(&std::env::temp_dir()).unwrap();
        let cache = temp.path().join("volume_helper_checked_v3.exe");
        ensure_helper(&cache, compile_helper, validate_helper, atomic_publish).unwrap();
        validate_helper(&cache).unwrap();
        ensure_helper(
            &cache,
            |_| panic!("valid executable should not recompile"),
            validate_helper,
            atomic_publish,
        )
        .unwrap();
        std::fs::write(&cache, b"interrupted old executable").unwrap();
        ensure_helper(&cache, compile_helper, validate_helper, atomic_publish).unwrap();
        validate_helper(&cache).unwrap();
    }
}

/// Find csc.exe from .NET Framework
fn find_csc() -> Result<PathBuf> {
    // Check common .NET Framework paths
    let framework_dir = PathBuf::from(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319");
    let csc = framework_dir.join("csc.exe");
    if csc.exists() {
        return Ok(csc);
    }

    // Try 32-bit
    let framework_dir = PathBuf::from(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319");
    let csc = framework_dir.join("csc.exe");
    if csc.exists() {
        return Ok(csc);
    }

    // Try to find via PATH
    let output = Command::new("where").arg("csc.exe").output();
    if let Ok(o) = output {
        if o.status.success() {
            let path = String::from_utf8_lossy(&o.stdout);
            let first_line = path.lines().next().unwrap_or("").trim();
            if !first_line.is_empty() {
                return Ok(PathBuf::from(first_line));
            }
        }
    }

    anyhow::bail!("csc.exe not found. .NET Framework 4.x is required for volume control.")
}

/// Get app data directory for storing helper binaries
fn get_app_data_dir() -> Result<PathBuf> {
    let appdata = std::env::var("APPDATA")
        .or_else(|_| std::env::var("USERPROFILE").map(|p| format!(r"{}\AppData\Roaming", p)))
        .context("Could not determine APPDATA directory")?;
    Ok(PathBuf::from(appdata).join("adaptive-controller"))
}
