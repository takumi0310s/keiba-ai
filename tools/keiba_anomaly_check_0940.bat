@echo off
:: ============================================================
:: Keiba-AnomalyCheck-0940 (DAILY 09:40 — 09:30 Discord 通知後 critical)
:: ★ 完成-1 sub-task (5/18) で作成 ★
::
:: 実 module: tools/anomaly_auto_detector.py
:: 注記: 仕様書 T6_anomaly_check.py --time 0940 は未存在、
::       実 module = anomaly_auto_detector.py、 --time arg なし
:: ============================================================
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
python -u tools\anomaly_auto_detector.py >> logs\keiba_anomaly_check_0940_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
