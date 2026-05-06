# register_jrdb_retry_schtasks.ps1
#
# JRDB AM 9:00 retry タスクを 土・日 で登録
# 06:00 の DailyJrdbKyi で 404 だった (TYB/SED/KYI 等) を retry
#
# Usage (admin 権限で):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_jrdb_retry_schtasks.ps1
#
# Rollback:
#   PowerShell -ExecutionPolicy Bypass -File tools\register_jrdb_retry_schtasks.ps1 -Rollback

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$BatPath = Join-Path $BaseDir "tools\jrdb_retry_am9.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "jrdb_retry_am9.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

$TaskNames = @("Keiba-JrdbRetryAm9_Sat", "Keiba-JrdbRetryAm9_Sun")

if ($Rollback) {
    Write-Host "=== Rollback: 2 task を削除 ==="
    foreach ($tn in $TaskNames) {
        if ($DryRun) {
            Write-Host "[DryRun] schtasks /Delete /TN $tn /F"
        } else {
            try {
                schtasks /Delete /TN $tn /F
                Write-Host "Deleted: $tn"
            } catch {
                Write-Warning "Skip (not found): $tn"
            }
        }
    }
    exit 0
}

Write-Host "=== JRDB AM 9:00 retry schtasks 登録 ==="

$Schedules = @(
    @{ Name = "Keiba-JrdbRetryAm9_Sat"; Day = "SAT"; },
    @{ Name = "Keiba-JrdbRetryAm9_Sun"; Day = "SUN"; }
)

foreach ($s in $Schedules) {
    $tn = $s.Name
    $day = $s.Day
    Write-Host ""
    Write-Host "--- $tn ($day 09:00) ---"

    $TR = "wscript.exe `"$VbsPath`" `"$BatPath`""

    if ($DryRun) {
        Write-Host "[DryRun] schtasks /Create /TN $tn /TR <略> /SC WEEKLY /D $day /ST 09:00 /F"
        continue
    }

    # 既存 task を削除 (idempotent)
    try {
        schtasks /Delete /TN $tn /F 2>$null | Out-Null
    } catch {}

    schtasks /Create `
        /TN $tn `
        /TR $TR `
        /SC WEEKLY `
        /D $day `
        /ST 09:00 `
        /RL LIMITED `
        /F

    if ($LASTEXITCODE -ne 0) {
        Write-Error "schtasks /Create 失敗: $tn"
        exit 1
    }
    Write-Host "Created: $tn"
}

Write-Host ""
Write-Host "=== 状態確認 ==="
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-JrdbRetryAm9*' } | Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "5/9 (土) 09:00 / 5/10 (日) 09:00 で TYB/SED/KYI/KAB の retry が自動発火"
Write-Host "ログ: logs\jrdb_retry_am9_YYYYMMDD.log"
Write-Host ""
Write-Host "Rollback: PowerShell -File $($MyInvocation.MyCommand.Path) -Rollback"
