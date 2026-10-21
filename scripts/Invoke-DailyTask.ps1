param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project environment missing. Run uv sync --locked first.'
}
# Resolve DATA_DIR using the same .env-aware configuration as the application.
$taskData = & $taskPython -c 'from video_agent.config import Settings; print(Settings.from_env().data_dir)'
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve project configuration.' }
New-Item -ItemType Directory -Path $taskData -Force | Out-Null
$taskLog = Join-Path $taskData 'daily-task.log'
& $taskPython -m video_agent.cli recommend *>> $taskLog
exit $LASTEXITCODE
