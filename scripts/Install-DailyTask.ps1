[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$TaskName = 'VideoAgent-Daily',
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')][string]$Time
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run uv sync --locked first.' }
if ((Get-TimeZone).Id -ne 'China Standard Time') {
    throw 'This local MVP task expects the Windows timezone China Standard Time (Asia/Shanghai).'
}
Push-Location -LiteralPath $taskRoot
try {
    $taskProfileJson = & $taskPython -m video_agent.cli profile
    if ($LASTEXITCODE -ne 0) { throw 'Unable to read profile.' }
    $taskProfile = ($taskProfileJson -join "`n") | ConvertFrom-Json
    if ($null -eq $taskProfile) { throw 'Initialize a profile before installing the daily task.' }
    if (-not $Time) { $Time = $taskProfile.daily_time }
} finally {
    Pop-Location
}
$taskScript = Join-Path $taskRoot 'scripts\Invoke-DailyTask.ps1'
$taskArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
    $taskScript + '" -RepositoryRoot "' + $taskRoot + '"'
$taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArguments -WorkingDirectory $taskRoot
$taskTrigger = New-ScheduledTaskTrigger -Daily -At $Time
$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
if ($PSCmdlet.ShouldProcess($TaskName, "Register daily recommendation at $Time Beijing time")) {
    Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $taskTrigger `
        -Settings $taskSettings -Description 'Local Video Agent daily learning recommendations'
}
