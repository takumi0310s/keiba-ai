@echo off
:: ============================================================
:: Keiba-AnomalyCheck-0830 (DAILY 08:30 — DailyPredict 後)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/anomaly_auto_detector.py
:: 注記: 仕様書 T6_anomaly_check.py --time 0830 は未存在、
::       実 module = anomaly_auto_detector.py、 --time arg なし
:: ============================================================
cd /d C:\Users\takum\keiba-ai
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
%PYTHON_EXE% -u tools\anomaly_auto_detector.py >> logs\keiba_anomaly_check_0830_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
