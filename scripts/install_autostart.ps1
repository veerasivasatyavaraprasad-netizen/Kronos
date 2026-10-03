# Start the Kronos live trader automatically every time you log in to Windows.
#   Right-click -> "Run with PowerShell"   (or:  powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1)
# Remove again with scripts\uninstall_autostart.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $repo "scripts\start_live.bat"

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "Kronos Live Trader" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Kronos real-time trading bot" -Force | Out-Null

Write-Host "Installed. The bot will start in its own window each time you log in."
Write-Host "Start it now with:  Start-ScheduledTask -TaskName 'Kronos Live Trader'"
