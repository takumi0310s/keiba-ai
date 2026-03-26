@echo off
REM Register daily premium scrape task (AM 03:00)
REM Run as Administrator

echo Setting up daily premium scrape task...
schtasks /create /tn "keiba-ai\DailyPremiumScrape" /tr "C:\Users\sato\keiba-ai\daily_premium_scrape.bat" /sc daily /st 03:00 /rl HIGHEST /f
if %errorlevel% equ 0 (
    echo Task created successfully!
    echo   Name: keiba-ai\DailyPremiumScrape
    echo   Schedule: Daily at 03:00
    echo   Script: C:\Users\sato\keiba-ai\daily_premium_scrape.bat
) else (
    echo Failed to create task. Run as Administrator.
)
pause
