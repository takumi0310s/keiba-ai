# register_strategy8_sidecar_schtasks.ps1
#
# 09:30 (土・日) で strategy8_sidecar.bat を発火する schtasks 登録。
#
# Strategy 8 sidecar:
# - V15 daily_predict (08:00) 完了後、 Jackpot pattern 該当馬を 別 Discord channel に通知
# - V15 production は 完全 不変
#
# Usage (admin 権限で):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_strategy8_sidecar_schtasks.ps1
#
# Rollback:
#   PowerShell -ExecutionPolicy Bypass -File tools\register_strategy8_sidecar_schtasks.ps1 -Rollback

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$BatPath = Join-Path $BaseDir "tools\strategy8_sidecar.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "strategy8_sidecar.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

$TaskNames = @("Keiba-Strategy8Sidecar_Sat", "Keiba-Strategy8Sidecar_Sun")

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

Write-Host "=== Step 1: Strategy 8 sidecar schtask 登録 ==="

$Schedules = @(
    @{ Name = "Keiba-Strategy8Sidecar_Sat"; Day = "SAT"; },
    @{ Name = "Keiba-Strategy8Sidecar_Sun"; Day = "SUN"; }
)

foreach ($s in $Schedules) {
    $tn = $s.Name
    $day = $s.Day
    Write-Host ""
    Write-Host "--- $tn ($day 09:30) ---"

    $TR = "wscript.exe `"$VbsPath`" `"$BatPath`""
    Write-Host "TR: $TR"

    if ($DryRun) {
        Write-Host "[DryRun] would register"
        continue
    }

    try {
        schtasks /Delete /TN $tn /F 2>$null | Out-Null
    } catch {}

    schtasks /Create `
        /TN $tn `
        /TR $TR `
        /SC WEEKLY `
        /D $day `
        /ST 09:30 `
        /RL LIMITED `
        /F

    if ($LASTEXITCODE -ne 0) {
        Write-Error "schtasks /Create 失敗: $tn"
        exit 1
    }
    Write-Host "Created: $tn"
}

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "5/16 (土) 09:30 / 5/17 (日) 09:30 で sidecar 自動発火。"
Write-Host "logs\strategy8_sidecar_YYYYMMDD.log"
