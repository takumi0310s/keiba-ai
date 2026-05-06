# register_multi_stage_predict_schtasks.ps1
#
# 当日 3 段階予測 schtasks 登録 (土日各 3 = 計 6 タスク)
#   - 10:00 Test10 (2R 馬体重補正 + 3R-12R 朝予測通知)
#   - 14:50 Race11_1450 (全 11R 予測、買い目は 1勝のみ)
#   - 15:45 Race12_1545 (全 12R 予測、案B改 1勝のみ買い目、主戦場)
#
# silent_runner.vbs 経由で静音化。
#
# Usage (admin 権限で):
#   PowerShell -ExecutionPolicy Bypass -File tools\register_multi_stage_predict_schtasks.ps1
#
# Rollback:
#   PowerShell -ExecutionPolicy Bypass -File tools\register_multi_stage_predict_schtasks.ps1 -Rollback

[CmdletBinding()]
param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\takum\keiba-ai"
$BatPath = Join-Path $BaseDir "tools\multi_stage_predict.bat"
$VbsPath = Join-Path $BaseDir "tools\silent_runner.vbs"

if (-not (Test-Path $BatPath)) {
    Write-Error "multi_stage_predict.bat not found: $BatPath"
    exit 1
}
if (-not (Test-Path $VbsPath)) {
    Write-Error "silent_runner.vbs not found: $VbsPath"
    exit 1
}

$Schedules = @(
    @{ Name = "Keiba-MultiStagePredict_Test10_Sat";       Day = "SAT"; Time = "10:00"; Stage = "test10" },
    @{ Name = "Keiba-MultiStagePredict_Test10_Sun";       Day = "SUN"; Time = "10:00"; Stage = "test10" },
    @{ Name = "Keiba-MultiStagePredict_Race11_1450_Sat";  Day = "SAT"; Time = "14:50"; Stage = "race11_1450" },
    @{ Name = "Keiba-MultiStagePredict_Race11_1450_Sun";  Day = "SUN"; Time = "14:50"; Stage = "race11_1450" },
    @{ Name = "Keiba-MultiStagePredict_Race12_1545_Sat";  Day = "SAT"; Time = "15:45"; Stage = "race12_1545" },
    @{ Name = "Keiba-MultiStagePredict_Race12_1545_Sun";  Day = "SUN"; Time = "15:45"; Stage = "race12_1545" }
)

if ($Rollback) {
    Write-Host "=== Rollback: 6 task を削除 ==="
    foreach ($s in $Schedules) {
        $tn = $s.Name
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

Write-Host "=== 当日 3 段階予測 schtasks 登録 (6 task) ==="

foreach ($s in $Schedules) {
    $tn = $s.Name
    $day = $s.Day
    $time = $s.Time
    $stg = $s.Stage
    Write-Host ""
    Write-Host "--- $tn ($day $time stage=$stg) ---"

    # TR: wscript silent_runner bat <stage>
    $TR = "wscript.exe `"$VbsPath`" `"$BatPath`" `"$stg`""

    if ($DryRun) {
        Write-Host "[DryRun] schtasks /Create /TN $tn /TR <略> /SC WEEKLY /D $day /ST $time /F"
        continue
    }

    # idempotent
    try {
        schtasks /Delete /TN $tn /F 2>$null | Out-Null
    } catch {}

    schtasks /Create `
        /TN $tn `
        /TR $TR `
        /SC WEEKLY `
        /D $day `
        /ST $time `
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
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-MultiStagePredict*' } | Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}} | Format-Table -AutoSize

Write-Host ""
Write-Host "=== 完了 ==="
Write-Host "5/9 (土) / 5/10 (日) で 10:00 / 14:50 / 15:45 に自動発火"
Write-Host "  - 10:00 → 2R 馬体重補正 + 3R-12R 朝予測通知 (情報提供)"
Write-Host "  - 14:50 → 11R 全予測 (重賞含む、買い目は 1勝クラスのみ)"
Write-Host "  - 15:45 → 12R 全予測 + 案B改 採用 R 買い目 (主戦場)"
Write-Host "ログ: logs\multi_stage_predict_<stage>_YYYYMMDD.log"
Write-Host ""
Write-Host "Rollback: PowerShell -File $($MyInvocation.MyCommand.Path) -Rollback"
