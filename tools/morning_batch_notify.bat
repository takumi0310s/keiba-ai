@echo off
rem 朝の一括通知 09:30 (2026-08-11 通知再整備。旧 RaceAutoNotify 08:45 の置換)
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
python tools\morning_batch_notify.py >> logs\morning_batch_notify_%TODAY%.log 2>&1
python tools\feature_health_report.py >> logs\feature_health_%TODAY%.log 2>&1
