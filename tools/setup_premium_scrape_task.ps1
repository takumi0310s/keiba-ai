# Windowsタスクスケジューラに毎日AM3:00の自動実行タスクを登録するPowerShellスクリプト
# 管理者権限で実行: powershell -ExecutionPolicy Bypass -File tools\setup_premium_scrape_task.ps1

$TaskName = "KeibaAI_DailyPremiumScrape"
$Description = "netkeiba premium data daily scraping (speed_index, newspaper, result)"
$BatPath = "C:\Users\sato\keiba-ai\daily_premium_scrape.bat"
$WorkDir = "C:\Users\sato\keiba-ai"

# 既存タスクがあれば削除
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Existing task removed."
}

# トリガー: 毎日AM3:00
$Trigger = New-ScheduledTaskTrigger -Daily -At "03:00"

# アクション: バッチファイル実行
$Action = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory $WorkDir

# 設定: PCスリープ中は起きたら実行、最大2時間
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

# ユーザー権限で登録（ログオン不要）
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description $Description `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Principal $Principal

Write-Host ""
Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Daily at 03:00"
Write-Host "  Action: $BatPath"
Write-Host ""
Write-Host "Verify with: Get-ScheduledTask -TaskName '$TaskName' | Format-List"
