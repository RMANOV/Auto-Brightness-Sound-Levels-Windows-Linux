//! Volume control
//!
//! Windows: Core Audio API via a pre-compiled C# helper.
//! The helper is compiled once from embedded source using csc.exe,
//! then reused for fast (<50ms) volume get/set operations.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;
use tracing::{debug, info, warn};

/// Embedded C# source for the volume helper executable.
/// COM interfaces must declare ALL methods in vtable order (after IUnknown's 3).
const VOLUME_HELPER_CS: &str = r#"
using System;
using System.Runtime.InteropServices;

// IMMDeviceEnumerator — vtable: IUnknown(3) + EnumAudioEndpoints, GetDefaultAudioEndpoint, ...
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int dwStateMask, out IntPtr ppDevices);
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppEndpoint);
    int GetDevice([MarshalAs(UnmanagedType.LPWStr)] string pwstrId, out IMMDevice ppDevice);
    int RegisterEndpointNotificationCallback(IntPtr pClient);
    int UnregisterEndpointNotificationCallback(IntPtr pClient);
}

// IMMDevice — vtable: IUnknown(3) + Activate, OpenPropertyStore, GetId, GetState
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
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
    int SetMasterVolumeLevelScalar(float fLevel, Guid pguidEventContext);
    int GetMasterVolumeLevel(out float pfLevelDB);
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
        enumerator.GetDefaultAudioEndpoint(0 /* eRender */, 1 /* eMultimedia */, out device);
        Guid iid = typeof(IAudioEndpointVolume).GUID;
        object activated;
        device.Activate(iid, 1 /* CLSCTX_ALL */, IntPtr.Zero, out activated);
        return (IAudioEndpointVolume)activated;
    }

    static int Main(string[] args) {
        if (args.Length < 1) {
            Console.Error.WriteLine("Usage: volume_helper.exe get | set <0-100>");
            return 1;
        }
        try {
            if (args[0] == "get") {
                float level;
                GetEndpointVolume().GetMasterVolumeLevelScalar(out level);
                Console.WriteLine(Math.Round(level * 100));
            } else if (args[0] == "set" && args.Length >= 2) {
                float pct;
                if (!float.TryParse(args[1], out pct)) { Console.Error.WriteLine("Invalid number"); return 1; }
                GetEndpointVolume().SetMasterVolumeLevelScalar(
                    Math.Max(0f, Math.Min(1f, pct / 100f)), Guid.Empty);
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

        let cs_path = app_dir.join("volume_helper.cs");
        let exe_path = app_dir.join("volume_helper.exe");

        // Compile helper if missing or source changed
        if !exe_path.exists() {
            info!("Compiling volume helper (one-time)...");
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

            if !status.status.success() {
                let stderr = String::from_utf8_lossy(&status.stderr);
                let stdout = String::from_utf8_lossy(&status.stdout);
                anyhow::bail!(
                    "C# compilation failed:\nstdout: {}\nstderr: {}",
                    stdout.trim(),
                    stderr.trim()
                );
            }
            info!("Volume helper compiled: {}", exe_path.display());
        }

        // Verify helper works
        let test = Command::new(&exe_path).arg("get").output();
        match test {
            Ok(o) if o.status.success() => {
                let vol = String::from_utf8_lossy(&o.stdout);
                info!("Volume control ready (current: {}%)", vol.trim());
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                warn!("Volume helper test returned error: {}", stderr.trim());
            }
            Err(e) => {
                warn!("Could not run volume helper: {}", e);
            }
        }

        Ok(Self {
            helper_path: exe_path,
        })
    }

    pub fn get(&self) -> Result<i32> {
        let output = Command::new(&self.helper_path)
            .arg("get")
            .output()
            .context("Failed to run volume helper")?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(val) = stdout.trim().parse::<f32>() {
                return Ok(val.round() as i32);
            }
        }

        warn!("Could not read volume, defaulting to 50%");
        Ok(50)
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
            warn!("Volume set failed: {}", stderr.trim());
        }

        Ok(())
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
