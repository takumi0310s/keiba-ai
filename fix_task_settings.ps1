# fix_task_settings.ps1
# Run as Administrator: powershell -ExecutionPolicy Bypass -File fix_task_settings.ps1
#
# Fixes Task Scheduler settings for keiba-ai tasks:
# - Allow start on batteries
# - Don't stop on batteries
# - Start when available (if missed)
# - Wake to run

$taskNames = @(
    "RaceAutoNotify_Sat",
    "RaceAutoNotify_Sun",
    "DailyPredict",
    "DailyPremiumScrape",
    "DailyResults_Sat",
    "DailyResults_Sun",
    "DailyResultsEvening",
    "WeeklyReport"
)

foreach ($name in $taskNames) {
    try {
        $task = Get-ScheduledTask -TaskPath "\keiba-ai\" -TaskName $name -ErrorAction Stop
        $settings = $task.Settings
        $settings.DisallowStartIfOnBatteries = $false
        $settings.StopIfGoingOnBatteries = $false
        $settings.StartWhenAvailable = $true
        # Wake only for race-critical tasks
        if ($name -like "RaceAutoNotify*" -or $name -eq "DailyPredict") {
            $settings.WakeToRun = $true
        }
        Set-ScheduledTask -TaskPath "\keiba-ai\" -TaskName $name -Settings $settings
        Write-Host "OK: $name" -ForegroundColor Green
    } catch {
        Write-Host "FAIL: $name - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. Verify with: Get-ScheduledTask -TaskPath '\keiba-ai\' | Select TaskName, State"
