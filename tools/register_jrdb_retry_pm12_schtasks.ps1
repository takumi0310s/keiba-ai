# register_jrdb_retry_pm12_schtasks.ps1
#
# JRDB PM 12:00 FINAL retry schtasks 登録 (土日)
# 06:00 DailyJrdbKyi + 09:00 JrdbRetryAm9 両方失敗時の最終手段
#
# Usage (admin):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_jrdb_retry_pm12_schtasks.ps1
# Rollback:
#   -Rollback option

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$BatPath = Join-Path $BaseDir "tools\jrdb_retry_pm12.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "jrdb_retry_pm12.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

$TaskNames = @("Keiba-JrdbRetryPm12_Sat", "Keiba-JrdbRetryPm12_Sun")

if ($Rollback) {
    Write-Host "=== Rollback ==="
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

Write-Host "=== JRDB PM 12:00 FINAL retry schtasks 登録 ==="

$Schedules = @(
    @{ Name = "Keiba-JrdbRetryPm12_Sat"; Day = "SAT"; },
    @{ Name = "Keiba-JrdbRetryPm12_Sun"; Day = "SUN"; }
)

foreach ($s in $Schedules) {
    $tn = $s.Name
    $day = $s.Day
    Write-Host ""
    Write-Host "--- $tn ($day 12:00) ---"

    $TR = "wscript.exe `"$VbsPath`" `"$BatPath`""

    if ($DryRun) {
        Write-Host "[DryRun] schtasks /Create /TN $tn /TR <略> /SC WEEKLY /D $day /ST 12:00 /F"
        continue
    }

    try { schtasks /Delete /TN $tn /F 2>$null | Out-Null } catch {}

    schtasks /Create `
        /TN $tn `
        /TR $TR `
        /SC WEEKLY `
        /D $day `
        /ST 12:00 `
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
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-JrdbRetryPm12*' } | Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "5/9 (土) / 5/10 (日) 12:00 で TYB/SED/KYI/KAB の FINAL retry"
Write-Host "失敗時は Discord で 'JRDB なしで投資判断' を yellow 通知"
Write-Host ""
Write-Host "Rollback: PowerShell -File $($MyInvocation.MyCommand.Path) -Rollback"
