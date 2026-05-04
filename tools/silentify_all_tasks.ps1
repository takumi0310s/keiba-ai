# silentify_all_tasks.ps1
# タスクスケジューラ全16件を wscript.exe + silent_runner.vbs 経由に切替え、
# ターミナルウィンドウを表示しないように一括変更する。
#
# 動作: backup JSON の各エントリについて、
#   旧: Execute=<bat>, Arguments=<orig>
#   新: Execute=wscript.exe, Arguments="silent_runner.vbs" "<bat>" <orig>
#
# 使い方 (管理者 PowerShell):
#   cd C:\Users\takum\keiba-ai
#   powershell -ExecutionPolicy Bypass -File tools\silentify_all_tasks.ps1
#
# rollback:
#   powershell -ExecutionPolicy Bypass -File tools\silentify_rollback.ps1

[CmdletBinding()]
param(
    [string]$BackupJson = "C:\Users\takum\keiba-ai\tools\task_silentify_backup_5_4.json",
    [string]$VbsPath    = "C:\Users\takum\keiba-ai\tools\silent_runner.vbs",
    [string]$LogPath    = "C:\Users\takum\keiba-ai\logs\silentify_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

# admin チェック
$current = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: 管理者権限が必要です。PowerShell を 管理者として実行 で起動し直してください。" -ForegroundColor Red
    exit 1
}

# log dir
$logDir = Split-Path $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# 前提ファイル
if (-not (Test-Path $BackupJson)) {
    Write-Host "ERROR: backup JSON が見つかりません: $BackupJson" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Host "ERROR: silent_runner.vbs が見つかりません: $VbsPath" -ForegroundColor Red
    exit 1
}

Write-Log "===== silentify_all_tasks 開始 ====="
Write-Log "BackupJson: $BackupJson"
Write-Log "VbsPath:    $VbsPath"

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

    # 既に静音化済みかチェック (Execute が wscript.exe で始まる)
    if ($oldExec -match 'wscript\.exe') {
        Write-Log "[$taskName] 既に静音化済み — skip"
        $success++
        continue
    }

    # bat path を抽出 (両端のダブルクォートを除去)
    $batPath = $oldExec.Trim().Trim('"')

    # 新 Arguments を構築 ("vbs path" "bat path" <orig args>)
    $newArgs = '"{0}" "{1}"' -f $VbsPath, $batPath
    if ($oldArgs -and $oldArgs.Trim().Length -gt 0) {
        $newArgs += " " + $oldArgs
    }

    Write-Log "[$taskName] 変更"
    Write-Log "  OLD Exec: $oldExec"
    Write-Log "  OLD Args: $oldArgs"
    Write-Log "  NEW Exec: wscript.exe"
    Write-Log "  NEW Args: $newArgs"

    try {
        # 既存 task の他属性 (WorkingDirectory) を保持
        $params = @{
            Execute   = "wscript.exe"
            Argument  = $newArgs
        }
        if ($oldWd) { $params.WorkingDirectory = $oldWd }

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
    Write-Host ""
    Write-Host "一部失敗しました。rollback したい場合は tools\silentify_rollback.ps1 を実行してください。" -ForegroundColor Yellow
    exit 2
}

Write-Log "全件成功。動作確認は次回スケジュール発火 (Keiba-TybPublishMonitor は毎時X:30) で確認可能。"
exit 0
