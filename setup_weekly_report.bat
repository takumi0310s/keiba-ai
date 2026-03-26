@echo off
REM 月曜AM8:00に週次レポートを自動実行
REM 管理者権限で実行すること

echo Setting up Weekly Report task...

schtasks /create /tn "keiba-ai\WeeklyReport" /tr "C:\Users\sato\keiba-ai\weekly_report.bat" /sc weekly /d MON /st 08:00 /rl HIGHEST /f

if %errorlevel% equ 0 (
    echo Task created:
    echo   keiba-ai\WeeklyReport - Monday 08:00
) else (
    echo Failed. Run as Administrator.
)
pause
