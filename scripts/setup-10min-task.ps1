# Adds the 10-minute auto brightness/volume task. The sunrise/sunset tasks stay;
# this one runs every 10 minutes between them, all day. Safe to re-run (updates).
# A run that overlaps a sunrise/sunset run exits at once (controller instance guard).
param(
    [string]$ControllerExe = "$env:USERPROFILE\Auto-Brightness-Sound-Levels-Windows-Linux\adaptive-rust\target\release\adaptive-controller.exe",
    [string]$TaskName = 'AdaptiveController-Every10Min'
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ControllerExe -PathType Leaf)) {
    throw "Controller not found: $ControllerExe - build it first: cargo build --release -p adaptive-controller --locked"
}
# The task starts in System32, so a relative path would never be found there.
$ControllerExe = (Resolve-Path -LiteralPath $ControllerExe).ProviderPath

# conhost --headless runs the console controller without opening a window.
# The controller ends its own run after 8 minutes: the task's time limit stops
# conhost, not the controller, and a stuck run must release the instance guard.
$conhost = Join-Path $env:SystemRoot 'System32\conhost.exe'
$action = New-ScheduledTaskAction -Execute $conhost `
    -Argument "--headless `"$ControllerExe`" --max-runtime-secs 480"

# Daily at 00:00, repeated every 10 minutes for the whole day.
$trigger = New-ScheduledTaskTrigger -Daily -At '00:00'
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At '00:00' `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 1)).Repetition

# The 9-minute limit is a backstop for conhost; overlapping starts are skipped.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 9)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'Auto brightness/volume every 10 minutes between the sunrise/sunset runs - Sofia' `
    -Force | Out-Null

Write-Host "Created: $TaskName (every 10 minutes, hidden window)"
