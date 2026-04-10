@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python tools/weekly_report.py >> logs\weekly_report.log 2>&1
