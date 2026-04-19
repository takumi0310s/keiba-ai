@echo off
chcp 65001 >nul
setlocal

REM process_watchdog v2 タスクスケジューラ用ランチャー（手動切替用）。
REM 既存 process_watchdog.bat は絶対に変更しないポリシーのため、こちらを並存。
REM 切り替え手順（手動）:
REM   schtasks /Change /TN "keiba-ai\ProcessWatchdog" /TR "C:\Users\takum\keiba-ai\tools\task_watchdog_v2.bat"

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\watchdog_v2_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] task_watchdog_v2 --once Start >> %LOGFILE%

python -u tools\process_watchdog_v2.py --once >> %LOGFILE% 2>&1

echo [%date% %time%] task_watchdog_v2 --once End >> %LOGFILE%
endlocal
