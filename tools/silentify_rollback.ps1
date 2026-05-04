# silentify_rollback.ps1
# 静音化を巻き戻す。backup JSON の Execute/Arguments/WorkingDirectory を
# そのまま Set-ScheduledTask で書き戻す。
#
# 使い方 (管理者 PowerShell):
#   powershell -ExecutionPolicy Bypass -File tools\silentify_rollback.ps1

[CmdletBinding()]
param(
    [string]$BackupJson = "C:\Users\takum\keiba-ai\tools\task_silentify_backup_5_4.json",
    [string]$LogPath    = "C:\Users\takum\keiba-ai\logs\silentify_rollback_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

$current = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: 管理者権限が必要です。" -ForegroundColor Red
    exit 1
}

$logDir = Split-Path $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

if (-not (Test-Path $BackupJson)) {
    Write-Host "ERROR: backup JSON が見つかりません: $BackupJson" -ForegroundColor Red
    exit 1
}

Write-Log "===== silentify_rollback 開始 ====="
$entries = Get-Content $BackupJson -Raw | ConvertFrom-Json
Write-Log "対象タスク件数: $($entries.Count)"

$success = 0
$failed  = @()

foreach ($e in $entries) {
    $taskName = $e.TaskName
    $taskPath = if ($e.TaskPath) { $e.TaskPath } else { "\" }
    $oldExec  = $e.Execute
    $oldArgs  = $e.Arguments
    $oldWd    = $e.WorkingDirectory

    Write-Log "[$taskName] rollback"
    Write-Log "  RESTORE Exec: $oldExec"
    Write-Log "  RESTORE Args: $oldArgs"

    try {
        $params = @{ Execute = $oldExec }
        if ($oldArgs)        { $params.Argument         = $oldArgs }
        if ($oldWd)          { $params.WorkingDirectory = $oldWd   }

        $action = New-ScheduledTaskAction @params
        Set-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Action $action | Out-Null
        Write-Log "  -> OK"
        $success++
    } catch {
        Write-Log "  -> FAILED: $($_.Exception.Message)"
        $failed += [PSCustomObject]@{ TaskName = $taskName; Error = $_.Exception.Message }
    }
}

Write-Log "===== 結果 ====="
Write-Log "成功: $success / $($entries.Count)"
if ($failed.Count -gt 0) {
    Write-Log "失敗: $($failed.Count)"
    foreach ($f in $failed) { Write-Log "  - $($f.TaskName): $($f.Error)" }
    exit 2
}
exit 0
