# Setup script — run once to create the new task architecture
# Removes old wait-based tasks, creates planner + direct controller tasks

# Remove old tasks
Unregister-ScheduledTask -TaskName 'AdaptiveController-Sunrise' -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'AdaptiveController-Sunset' -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'AdaptiveController-Planner' -Confirm:$false -ErrorAction SilentlyContinue

# Create daily planner task (runs at logon + midnight, completely hidden)
$plannerScript = "$env:APPDATA\adaptive-controller\daily-planner.ps1"
$plannerAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$plannerScript`""

$triggerMidnight = New-ScheduledTaskTrigger -Daily -At '00:05'

$plannerSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask -TaskName 'AdaptiveController-Planner' `
    -Action $plannerAction `
    -Trigger $triggerMidnight `
    -Settings $plannerSettings `
    -Description 'Daily: calculates sunrise/sunset for Sofia, sets exact trigger times' `
    -Force | Out-Null

Write-Host "Created: AdaptiveController-Planner (daily 00:05)"

# Run the planner now to create today's sunrise/sunset tasks
Write-Host "Running planner for today..."
& $plannerScript

# Show all tasks
Write-Host ""
Write-Host "=== Scheduled Tasks ==="
Get-ScheduledTask -TaskName 'AdaptiveController-*' | ForEach-Object {
    $info = Get-ScheduledTaskInfo -InputObject $_
    $next = if ($info.NextRunTime -and $info.NextRunTime -gt (Get-Date "2000-01-01")) {
        $info.NextRunTime.ToString("yyyy-MM-dd HH:mm")
    } else { "on-demand" }
    Write-Host ("  {0,-35} State={1,-8} Next={2}" -f $_.TaskName, $_.State, $next)
}
