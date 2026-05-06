# register_morning_weight_check_schtasks.ps1
#
# 09:30 (土・日) で morning_weight_check.bat を発火する schtasks 登録。
# silent_runner.vbs 経由で静音化。
#
# 監視対象: 案B改 採用候補 (12R 1勝クラス、trio/umaren)
# 想定実行時間: 案B改 0-3 R x 30 秒 = 1-2 分
#
# Usage (admin 権限で):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_morning_weight_check_schtasks.ps1
#
# Rollback:
#   PowerShell -ExecutionPolicy Bypass -File tools\register_morning_weight_check_schtasks.ps1 -Rollback

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$BatPath = Join-Path $BaseDir "tools\morning_weight_check.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "morning_weight_check.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

$TaskNames = @("Keiba-MorningWeightCheck_Sat", "Keiba-MorningWeightCheck_Sun")

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

Write-Host "=== Step 1: 既存 task の確認 (なければ新規) ==="

# 09:30 の trigger を 土・日 別々に設定
$Schedules = @(
    @{ Name = "Keiba-MorningWeightCheck_Sat"; Day = "SAT"; },
    @{ Name = "Keiba-MorningWeightCheck_Sun"; Day = "SUN"; }
)

foreach ($s in $Schedules) {
    $tn = $s.Name
    $day = $s.Day
    Write-Host ""
    Write-Host "--- $tn ($day 09:30) ---"

    # TR: wscript.exe + silent_runner + bat
    $TR = "wscript.exe `"$VbsPath`" `"$BatPath`""
    Write-Host "TR: $TR"

    if ($DryRun) {
        Write-Host "[DryRun] schtasks /Create /TN $tn /TR <略> /SC WEEKLY /D $day /ST 09:30 /F"
        continue
    }

    # 既存 task を削除 (idempotent)
    try {
        schtasks /Delete /TN $tn /F 2>$null | Out-Null
    } catch {}

    # 新規作成
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
Write-Host "=== Step 2: 状態確認 ==="
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-MorningWeightCheck*' } | Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "5/9 (土) 09:30 / 5/10 (日) 09:30 で自動発火します。"
Write-Host "ログ: logs\morning_weight_check_YYYYMMDD.log"
Write-Host ""
Write-Host "Rollback: PowerShell -File $($MyInvocation.MyCommand.Path) -Rollback"
