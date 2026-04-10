@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python tools/daily_predict.py >> logs\daily_predict.log 2>&1
