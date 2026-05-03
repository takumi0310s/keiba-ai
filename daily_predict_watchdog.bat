@echo off
REM daily_predict.py を watchdog 経由で実行 (中断検知+自動再起動+Discord通知)
REM Usage: daily_predict_watchdog.bat
REM
REM 既存 daily_predict.bat と置き換える代替。
REM タスクスケジューラ DailyPredict のコマンドを本 .bat に変更すれば
REM 5/4 以降は中断検知・自動再起動が機能する。

setlocal
chcp 65001 >nul
cd /d C:\Users\takum\keiba-ai

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
set LOGFILE=logs\daily_predict_watchdog_wrapper_%TODAY%.log

echo [%date% %time%] daily_predict_watchdog START (TODAY=%TODAY%) >> %LOGFILE%
python tools\daily_predict_watchdog.py --date %TODAY% >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] daily_predict_watchdog END exitcode=%EXITCODE% >> %LOGFILE%
exit /b %EXITCODE%
