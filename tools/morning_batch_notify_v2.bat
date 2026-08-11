@echo off
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
python tools\morning_batch_notify.py --with-health >> logs\morning_batch_notify_%TODAY%.log 2>&1
python tools\feature_health_report.py --save-only >> logs\feature_health_%TODAY%.log 2>&1
