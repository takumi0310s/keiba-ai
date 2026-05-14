# 5/16 (土) Enhanced 機能 schtask 登録 script
# user admin 権限 で 実行:
#   powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_5_16_enhancement_schtasks.ps1
#
# 登録 task:
# 1. Keiba-DangerHorseAlert: 09:00 (土日) - 危険 horse Discord 通知
# 2. Strategy 8 sidecar は 既存 register_strategy8_sidecar_schtasks.ps1 で 別途 登録

$ErrorActionPreference = "Stop"

# Keiba-DangerHorseAlert (09:00 土日)
$TaskName1 = "Keiba-DangerHorseAlert"
$Action1 = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c C:\Users\takum\keiba-ai\tools\danger_horse_alert.bat"
$Trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At 09:00
$Settings1 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries

try {
    Register-ScheduledTask -TaskName $TaskName1 -Action $Action1 -Trigger $Trigger1 `
        -Settings $Settings1 -RunLevel Limited -Force
    Write-Host "[OK] $TaskName1 登録 (土日 09:00)"
} catch {
    Write-Host "[ERROR] $TaskName1 登録失敗: $_"
}

Write-Host ""
Write-Host "=== 5/16 ready task 一覧 ==="
Get-ScheduledTask -TaskName "Keiba-*" | ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    Write-Host ("  - {0}: state={1}, next={2}" -f $_.TaskName, $_.State, $info.NextRunTime)
}
