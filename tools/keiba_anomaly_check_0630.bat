@echo off
:: ============================================================
:: Keiba-AnomalyCheck-0630 (DAILY 06:30 — DailyJrdbKyi 後)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/anomaly_auto_detector.py
:: 注記: 仕様書 T6_anomaly_check.py --time 0630 は未存在、
::       実 module = anomaly_auto_detector.py、 --time arg なし
::       (時刻別の差別化は schtask /ST のみで実現)
:: ============================================================
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
python -u tools\anomaly_auto_detector.py >> logs\keiba_anomaly_check_0630_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
