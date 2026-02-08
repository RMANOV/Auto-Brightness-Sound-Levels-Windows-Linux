# Adaptive Controller — Daily Planner
# Runs at boot/login + midnight. Calculates today's sunrise/sunset for Sofia,
# then updates the two scheduled task triggers to the EXACT times.
# No waiting, no visible windows. Completes in < 1 second.

$Latitude = 42.6977
$Longitude = 23.3219
$TimezoneOffset = 2.0  # UTC+2 (EET); change to 3.0 for EEST (summer)

$ControllerExe = "$env:USERPROFILE\Auto-Brightness-Sound-Levels-Windows-Linux\adaptive-rust\target\release\adaptive-controller.exe"
$LogFile = "$env:APPDATA\adaptive-controller\controller.log"

function Get-SunTimes {
    param([int]$Year, [int]$Month, [int]$Day, [double]$Lat, [double]$Lon, [double]$TZ)

    $a = [int]((14 - $Month) / 12)
    $y = $Year + 4800 - $a
    $m = $Month + 12 * $a - 3
    $JD = $Day + [Math]::Floor((153 * $m + 2) / 5) + 365 * $y +
          [Math]::Floor($y / 4) - [Math]::Floor($y / 100) + [Math]::Floor($y / 400) - 32045

    $T = ($JD - 2451545.0) / 36525.0
    $M = 357.52911 + $T * (35999.05029 - 0.0001537 * $T)
    $Mrad = $M * [Math]::PI / 180.0
    $C = [Math]::Sin($Mrad) * (1.914602 - $T * (0.004817 + 0.000014 * $T)) +
         [Math]::Sin(2 * $Mrad) * (0.019993 - 0.000101 * $T) +
         [Math]::Sin(3 * $Mrad) * 0.000289
    $L0 = 280.46646 + $T * (36000.76983 + 0.0003032 * $T)
    $SunTrueLon = $L0 + $C
    $Omega = 125.04 - 1934.136 * $T
    $Lambda = $SunTrueLon - 0.00569 - 0.00478 * [Math]::Sin($Omega * [Math]::PI / 180.0)
    $E0 = 23.0 + (26.0 + (21.448 - $T * (46.815 + $T * (0.00059 - $T * 0.001813))) / 60.0) / 60.0
    $Eps = $E0 + 0.00256 * [Math]::Cos($Omega * [Math]::PI / 180.0)
    $SinDec = [Math]::Sin($Eps * [Math]::PI / 180.0) * [Math]::Sin($Lambda * [Math]::PI / 180.0)
    $Dec = [Math]::Asin($SinDec) * 180.0 / [Math]::PI
    $EpsRad = $Eps * [Math]::PI / 180.0
    $L0Rad = $SunTrueLon * [Math]::PI / 180.0
    $E = 0.016708634 - $T * (0.000042037 + 0.0000001267 * $T)
    $Y = [Math]::Pow([Math]::Tan($EpsRad / 2.0), 2)
    $EqTime = $Y * [Math]::Sin(2 * $L0Rad) - 2 * $E * [Math]::Sin($Mrad) +
              4 * $E * $Y * [Math]::Sin($Mrad) * [Math]::Cos(2 * $L0Rad) -
              0.5 * $Y * $Y * [Math]::Sin(4 * $L0Rad) -
              1.25 * $E * $E * [Math]::Sin(2 * $Mrad)
    $EqTimeMin = ($EqTime * 180.0 / [Math]::PI) * 4.0
    $HAarg = [Math]::Cos(90.833 * [Math]::PI / 180.0) /
             ([Math]::Cos($Lat * [Math]::PI / 180.0) * [Math]::Cos($Dec * [Math]::PI / 180.0)) -
             [Math]::Tan($Lat * [Math]::PI / 180.0) * [Math]::Tan($Dec * [Math]::PI / 180.0)
    $HA = [Math]::Acos([Math]::Max(-1, [Math]::Min(1, $HAarg))) * 180.0 / [Math]::PI
    $Correction = $EqTimeMin + 4.0 * $Lon
    $NoonUTC = 720.0 - $Correction
    $Offset = $TZ * 60.0
    $Sunrise = (($NoonUTC - 4.0 * $HA + $Offset) % 1440 + 1440) % 1440
    $Sunset  = (($NoonUTC + 4.0 * $HA + $Offset) % 1440 + 1440) % 1440
    return @{ Sunrise = $Sunrise; Sunset = $Sunset }
}

# --- Main ---

$Now = Get-Date
$Sun = Get-SunTimes -Year $Now.Year -Month $Now.Month -Day $Now.Day `
                    -Lat $Latitude -Lon $Longitude -TZ $TimezoneOffset

$SrTime = (Get-Date).Date.AddMinutes($Sun.Sunrise - 30)  # 30 min before sunrise
$SsTime = (Get-Date).Date.AddMinutes($Sun.Sunset  - 30)  # 30 min before sunset

$SrStr = $SrTime.ToString("HH:mm")
$SsStr = $SsTime.ToString("HH:mm")
$SrEvent = (Get-Date).Date.AddMinutes($Sun.Sunrise).ToString("HH:mm")
$SsEvent = (Get-Date).Date.AddMinutes($Sun.Sunset).ToString("HH:mm")

$msg = "[{0}] Planner: sunrise {1} (run {2}), sunset {3} (run {4})" -f `
       (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $SrEvent, $SrStr, $SsEvent, $SsStr
Add-Content -Path $LogFile -Value $msg

# Update sunrise task trigger to exact time
$action = New-ScheduledTaskAction -Execute $ControllerExe
$triggerSr = New-ScheduledTaskTrigger -Once -At $SrTime
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$descSr = "Auto brightness/volume at sunrise ($SrEvent) - Sofia"
Register-ScheduledTask -TaskName 'AdaptiveController-Sunrise' `
    -Action $action -Trigger $triggerSr -Settings $settings `
    -Description $descSr -Force | Out-Null

# Update sunset task trigger to exact time
$triggerSs = New-ScheduledTaskTrigger -Once -At $SsTime
$descSs = "Auto brightness/volume at sunset ($SsEvent) - Sofia"
Register-ScheduledTask -TaskName 'AdaptiveController-Sunset' `
    -Action $action -Trigger $triggerSs -Settings $settings `
    -Description $descSs -Force | Out-Null
