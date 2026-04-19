# Keiba-AM3FireCheck タスクスケジューラ登録スクリプト
#
# 毎日 03:15 に tools/am3_fire_check.py を実行し、AM3:00 DailyPremiumScrape の
# 発火結果を Discord に自動通知する Reverse-Watchdog。
#
# 実行:
#   powershell -ExecutionPolicy Bypass -File tools/task_register_am3_firecheck.ps1
#
# アンインストール:
#   schtasks /delete /tn Keiba-AM3FireCheck /f

$ErrorActionPreference = "Stop"

$TaskName = "Keiba-AM3FireCheck"
$BatPath = "C:\Users\takum\keiba-ai\am3_fire_check.bat"

$action = New-ScheduledTaskAction `
    -Execute $BatPath `
    -WorkingDirectory "C:\Users\takum\keiba-ai"

$trigger = New-ScheduledTaskTrigger -Daily -At "03:15"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Force `
    | Out-Null

Write-Host "Registered: $TaskName (daily 03:15)"
Get-ScheduledTask -TaskName $TaskName | Format-List TaskName, State
