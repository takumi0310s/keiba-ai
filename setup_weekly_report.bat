@echo off
REM Register weekly report task (Monday 08:00)
REM Run as Administrator

echo Setting up Weekly Report task...

schtasks /create /tn "keiba-ai\WeeklyReport" /tr "C:\Users\sato\keiba-ai\weekly_report.bat" /sc weekly /d MON /st 08:00 /rl HIGHEST /f

if %errorlevel% equ 0 (
    echo Task created:
    echo   keiba-ai\WeeklyReport - Monday 08:00
) else (
    echo Failed. Run as Administrator.
)
pause
