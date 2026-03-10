@echo off
cd /d C:\Users\sato\keiba-ai
python tools/daily_results.py >> logs\daily_results.log 2>&1
