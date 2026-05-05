# register_nar_schtasks.ps1
# NAR 関連 schtasks 登録 (admin 必要)
#
# 登録 task (5 件):
#   Keiba-NarMidDayCalendar  13:00 daily  (将来 scrape_nar_calendar)
#   Keiba-NarDailyScrape     16:30 daily  (将来 scrape_nar_today)
#   Keiba-NarDailyPredict    17:00 daily  (現実装 nar_daily_pipeline.bat)
#   Keiba-NarLiveOddsRefresh 19:00 daily  (将来 odds 再取得)
#   Keiba-NarDailyResults    21:30 daily  (将来 結果照合)
#
# 既存 task と時刻衝突なし (DailyPredict 08:00, NightlySanity 23:00 等)
# silent_runner.vbs 経由で hidden 起動 (静音化済)
#
# Usage (admin PowerShell):
#   cd C:\Users\takum\keiba-ai
#   powershell -ExecutionPolicy Bypass -File tools\register_nar_schtasks.ps1

[CmdletBinding()]
param(
    [string]$BaseDir = "C:\Users\takum\keiba-ai",
    [string]$VbsPath = "C:\Users\takum\keiba-ai\tools\silent_runner.vbs",
    [string]$LogPath = "C:\Users\takum\keiba-ai\logs\register_nar_schtasks_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
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

$logDir = Split-Path $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

Write-Log "===== register_nar_schtasks 開始 ====="
Write-Log "BaseDir: $BaseDir"
Write-Log "VbsPath: $VbsPath"

if (-not (Test-Path $VbsPath)) {
    Write-Host "ERROR: silent_runner.vbs が見つかりません: $VbsPath" -ForegroundColor Red
    exit 1
}

# 各 task 定義 (現時点で実装済は NarDailyPredict のみ。他は placeholder で同 bat 指定、将来 script 追加時に Set-ScheduledTask で書換え)
$NarDailyBat = "$BaseDir\tools\nar_daily_pipeline.bat"
if (-not (Test-Path $NarDailyBat)) {
    Write-Host "ERROR: nar_daily_pipeline.bat が見つかりません: $NarDailyBat" -ForegroundColor Red
    exit 1
}

$tasks = @(
    @{ Name="Keiba-NarMidDayCalendar"; Time="13:00"; Bat=$NarDailyBat; Desc="NAR 当日カレンダー取得 (placeholder, 将来 scrape_nar_calendar)" },
    @{ Name="Keiba-NarDailyScrape";    Time="16:30"; Bat=$NarDailyBat; Desc="NAR 当日出馬表 + 前夜オッズ (placeholder, 将来 scrape_nar_today)" },
    @{ Name="Keiba-NarDailyPredict";   Time="17:00"; Bat=$NarDailyBat; Desc="NAR 推論 + 候補抽出" },
    @{ Name="Keiba-NarLiveOddsRefresh"; Time="19:00"; Bat=$NarDailyBat; Desc="NAR live odds (placeholder, 将来 race 単位 refresh)" },
    @{ Name="Keiba-NarDailyResults";   Time="21:30"; Bat=$NarDailyBat; Desc="NAR 結果照合 (placeholder, 将来 nar_daily_results)" }
)

$success = 0
$failed = @()

foreach ($t in $tasks) {
    Write-Log "[$($t.Name)] 登録"
    Write-Log "  Schedule: DAILY $($t.Time)"
    Write-Log "  Bat: $($t.Bat)"
    Write-Log "  Desc: $($t.Desc)"

    try {
        # 既存 task 削除 (再登録)
        $existing = Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Log "  既存 task あり → 削除 → 再作成"
            Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
        }

        # 新規作成
        $action = New-ScheduledTaskAction `
            -Execute "wscript.exe" `
            -Argument "`"$VbsPath`" `"$($t.Bat)`"" `
            -WorkingDirectory $BaseDir
        $trigger = New-ScheduledTaskTrigger -Daily -At $t.Time
        $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::FromHours(1))
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

        Register-ScheduledTask `
            -TaskName $t.Name `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description $t.Desc | Out-Null

        Write-Log "  -> OK"
        $success++
    } catch {
        Write-Log "  -> FAILED: $($_.Exception.Message)"
        $failed += [PSCustomObject]@{ Name=$t.Name; Error=$_.Exception.Message }
    }
}

Write-Log "===== 結果 ====="
Write-Log "成功: $success / $($tasks.Count)"
if ($failed.Count -gt 0) {
    Write-Log "失敗: $($failed.Count)"
    foreach ($f in $failed) { Write-Log "  - $($f.Name): $($f.Error)" }
    exit 2
}

Write-Log "全件登録完了。Get-ScheduledTask -TaskName 'Keiba-Nar*' で一覧確認可能。"
exit 0
