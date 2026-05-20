@echo off
:: ============================================================
:: Keiba-CumulativeAudit (DAILY 21:00)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/daily_cumulative_audit.py
:: DailyResultsEvening (20:00) の 1 時間後想定、 read-only drift detect
:: 別途 tools/daily_cumulative_audit.bat (既存) も存在するが、
:: 完成-1 仕様書名 = keiba_cumulative_audit.bat を新規作成
:: ============================================================
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe
%PYTHON_EXE% -u tools\daily_cumulative_audit.py >> logs\keiba_cumulative_audit_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
