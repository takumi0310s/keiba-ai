@echo off
:: ============================================================
:: Keiba-AnomalyCheck-1700 (DAILY 17:00 — G1 後 evening 整理 check)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/anomaly_auto_detector.py
:: 注記: 仕様書 T6_anomaly_check.py --time 1700 は未存在、
::       実 module = anomaly_auto_detector.py、 --time arg なし
:: ============================================================
cd /d C:\Users\takum\keiba-ai
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
%PYTHON_EXE% -u tools\anomaly_auto_detector.py >> logs\keiba_anomaly_check_1700_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
