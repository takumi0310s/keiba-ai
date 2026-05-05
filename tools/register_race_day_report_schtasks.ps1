# register_race_day_report_schtasks.ps1
# 当日結果サマリー自動レポート用 schtasks 登録 (admin 必要)
#
# 登録 task (2 件):
#   Keiba-RaceDayReport_Sat   土曜 18:00 daily (土曜のみ条件は WeeklyTrigger)
#   Keiba-RaceDayReport_Sun   日曜 18:00 daily
#
# silent_runner.vbs 経由で hidden 起動 (静音化済)
# 既存 DailyResultsEvening (20:00) と被らない。 18:00 は DailyResults_Sat と同時刻 → 既存と分け、
# DailyResults_Sat 後 (15-30 分後) で発火させるため 18:30 に変更可。本実装は 18:00 (parallel OK)。
#
# Usage (admin PowerShell):
#   cd C:\Users\takum\keiba-ai
#   powershell -ExecutionPolicy Bypass -File tools\register_race_day_report_schtasks.ps1

[CmdletBinding()]
param(
    [string]$BaseDir = "C:\Users\takum\keiba-ai",
    [string]$VbsPath = "C:\Users\takum\keiba-ai\tools\silent_runner.vbs",
    [string]$LogPath = "C:\Users\takum\keiba-ai\logs\register_race_day_report_schtasks_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
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
    Write-Host "ERROR: 管理者権限が必要です。" -ForegroundColor Red
    exit 1
}

$logDir = Split-Path $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

if (-not (Test-Path $VbsPath)) {
    Write-Host "ERROR: silent_runner.vbs not found: $VbsPath" -ForegroundColor Red
    exit 1
}

# bat wrapper
$BatPath = "$BaseDir\race_day_report.bat"
if (-not (Test-Path $BatPath)) {
    @"
@echo off
cd /d $BaseDir
set LOG=$BaseDir\logs\race_day_report_%DATE:/=%.log
python tools\race_day_report.py >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $BatPath -Encoding ASCII
    Write-Log "[INFO] $BatPath を新規作成"
}

Write-Log "===== register_race_day_report_schtasks 開始 ====="

$tasks = @(
    @{ Name="Keiba-RaceDayReport_Sat"; DayOfWeek="Saturday"; Time="18:00" },
    @{ Name="Keiba-RaceDayReport_Sun"; DayOfWeek="Sunday";   Time="18:00" }
)

$success = 0
$failed = @()

foreach ($t in $tasks) {
    Write-Log "[$($t.Name)] 登録 (週次 $($t.DayOfWeek) $($t.Time))"
    try {
        $existing = Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Log "  既存あり → 削除 → 再作成"
            Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
        }

        $action = New-ScheduledTaskAction `
            -Execute "wscript.exe" `
            -Argument "`"$VbsPath`" `"$BatPath`"" `
            -WorkingDirectory $BaseDir
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $t.DayOfWeek -At $t.Time
        $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::FromMinutes(20))
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

        Register-ScheduledTask `
            -TaskName $t.Name `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description "race_day_report 当日結果サマリー Discord 通知" | Out-Null

        Write-Log "  -> OK"
        $success++
    } catch {
        Write-Log "  -> FAILED: $($_.Exception.Message)"
        $failed += [PSCustomObject]@{ Name=$t.Name; Error=$_.Exception.Message }
    }
}

Write-Log "===== 結果: $success / $($tasks.Count) ====="
if ($failed.Count -gt 0) {
    foreach ($f in $failed) { Write-Log "  - $($f.Name): $($f.Error)" }
    exit 2
}
exit 0
