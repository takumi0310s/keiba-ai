@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python tools/daily_results.py >> logs\daily_results.log 2>&1
