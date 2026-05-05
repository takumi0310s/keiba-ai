# register_process_watchdog_v2.ps1
#
# 既存 ProcessWatchdog (Disabled, v1) を v2 (ログ鮮度ベース) に切替えて Enable。
# silent_runner.vbs 経由で静音化して 5 分間隔で発火。
#
# 監視対象:
#   - daily_predict      : logs/daily_predict*.log が 30分以上更新なしで DEAD
#   - race_auto_notify   : logs/race_auto_notify*.log が 10分以上更新なしで DEAD
#
# 再起動ポリシー: 07:00-18:00 のみ再起動、それ以外は Discord 警告通知のみ
#
# Usage (admin 権限で):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1
#
# Rollback (元に戻す場合):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1 -Rollback

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$TaskName = "ProcessWatchdog"
$BatPath = Join-Path $BaseDir "tools\task_watchdog_v2.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "task_watchdog_v2.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

if ($Rollback) {
    Write-Host "=== Rollback: ProcessWatchdog を Disable に戻す ==="
    if ($DryRun) {
        Write-Host "[DryRun] schtasks /Change /TN $TaskName /DISABLE"
    } else {
        schtasks /Change /TN $TaskName /DISABLE
        Write-Host "Done. ProcessWatchdog disabled."
    }
    exit 0
}

Write-Host "=== Step 1: 既存タスク確認 ==="
$existing = Get-ScheduledTask | Where-Object { $_.TaskName -eq $TaskName } | Select-Object -First 1
if (-not $existing) {
    Write-Error "ProcessWatchdog タスクが見つかりません。新規登録が必要なら別途 schtasks /Create コマンドで作成してください。"
    exit 1
}
Write-Host "Found: $TaskName (State=$($existing.State))"

Write-Host ""
Write-Host "=== Step 2: TR (実行コマンド) を v2 に切替 ==="
# wscript.exe silent_runner.vbs task_watchdog_v2.bat で静音化
$NewTR = "wscript.exe `"$VbsPath`" `"$BatPath`""
Write-Host "新 TR: $NewTR"

if ($DryRun) {
    Write-Host "[DryRun] schtasks /Change /TN $TaskName /TR `"$NewTR`""
} else {
    schtasks /Change /TN $TaskName /TR $NewTR
    if ($LASTEXITCODE -ne 0) {
        Write-Error "schtasks /Change 失敗 (TR 変更)"
        exit 1
    }
    Write-Host "TR 変更 OK"
}

Write-Host ""
Write-Host "=== Step 3: トリガ確認 (5分間隔) ==="
# 既存トリガが 5分間隔でなければ警告
$triggers = (Get-ScheduledTask -TaskName $TaskName).Triggers
foreach ($t in $triggers) {
    Write-Host "  Trigger: $($t.GetType().Name) Repetition=$($t.Repetition.Interval)"
}

Write-Host ""
Write-Host "=== Step 4: タスク Enable ==="
if ($DryRun) {
    Write-Host "[DryRun] schtasks /Change /TN $TaskName /ENABLE"
} else {
    schtasks /Change /TN $TaskName /ENABLE
    if ($LASTEXITCODE -ne 0) {
        Write-Error "schtasks /Change /ENABLE 失敗"
        exit 1
    }
    Write-Host "Enable OK"
}

Write-Host ""
Write-Host "=== Step 5: 動作確認 (--once 1回手動実行) ==="
if ($DryRun) {
    Write-Host "[DryRun] python tools\process_watchdog_v2.py --once --dry-run"
} else {
    Push-Location $BaseDir
    try {
        & python "tools\process_watchdog_v2.py" "--once" "--dry-run"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "動作確認 OK"
        } else {
            Write-Warning "watchdog_v2 --once --dry-run が exit code $LASTEXITCODE を返しました。logs\ を確認してください"
        }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "=== Step 6: 状態最終確認 ==="
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State, @{N='Action';E={$_.Actions[0].Execute + ' ' + $_.Actions[0].Arguments}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "次回トリガで logs\watchdog_v2_YYYYMMDD.log にログが記録されます。"
Write-Host ""
Write-Host "Rollback したい場合:"
Write-Host "  PowerShell -File $($MyInvocation.MyCommand.Path) -Rollback"
