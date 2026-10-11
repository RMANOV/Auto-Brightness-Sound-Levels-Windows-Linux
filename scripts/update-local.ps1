# One command to bring a local Windows install up to date: pull main, rebuild the
# controller, re-register the scheduled tasks (planner, sunrise/sunset and the
# 10-minute task) and start a run with the new code now.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\update-local.ps1
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot

function Step([string]$Text) { Write-Host "`n==> $Text" }
function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

Step "Updating $Repo from origin/main"
Invoke-Native 'git fetch' { git -C $Repo fetch origin main }
if ((git -C $Repo rev-parse --abbrev-ref HEAD) -ne 'main') {
    Invoke-Native 'git checkout main' { git -C $Repo checkout main }
}
Invoke-Native 'git merge' { git -C $Repo merge --ff-only origin/main }

# Windows locks a running .exe, so the build could not replace it. Pause the
# tasks first so no scheduled run starts the old controller during the build.
Step 'Pausing the scheduled tasks and stopping the running controller'
$tasks = @(Get-ScheduledTask -TaskName 'AdaptiveController-*' -ErrorAction SilentlyContinue |
    Where-Object State -ne 'Disabled')
$tasks | Disable-ScheduledTask | Out-Null
try {
    $running = @(Get-Process -Name 'adaptive-controller' -ErrorAction SilentlyContinue)
    if ($running.Count) {
        $running | Stop-Process -Force
        $running | Wait-Process -Timeout 15 -ErrorAction SilentlyContinue
    }

    Step 'Building the release controller'
    Push-Location (Join-Path $Repo 'adaptive-rust')
    try {
        Invoke-Native 'cargo build' { cargo build --release -p adaptive-controller --locked }
    } finally {
        Pop-Location
    }
} finally {
    $tasks | Enable-ScheduledTask | Out-Null
}

# setup-tasks.ps1 runs the planner from %APPDATA%. Install it there when missing,
# but never overwrite a copy the user edited (timezone, coordinates, exe path).
Step 'Registering the scheduled tasks'
$plannerDir = Join-Path $env:APPDATA 'adaptive-controller'
$plannerSrc = Join-Path $PSScriptRoot 'daily-planner.ps1'
$plannerDest = Join-Path $plannerDir 'daily-planner.ps1'
New-Item -ItemType Directory -Force -Path $plannerDir | Out-Null
if (-not (Test-Path -LiteralPath $plannerDest)) {
    Copy-Item -LiteralPath $plannerSrc -Destination $plannerDest
} elseif ((Get-FileHash -LiteralPath $plannerDest).Hash -ne (Get-FileHash -LiteralPath $plannerSrc).Hash) {
    Write-Warning "Kept your edited $plannerDest. To use the repository version: Copy-Item '$plannerSrc' '$plannerDest' -Force"
}
& (Join-Path $PSScriptRoot 'setup-tasks.ps1')

Step 'Starting a run with the new code'
Start-ScheduledTask -TaskName 'AdaptiveController-Every10Min'
Write-Host 'Done: the new controller runs now and then every 10 minutes, plus the sunrise/sunset runs.'
