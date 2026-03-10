@echo off
cd /d C:\Users\sato\keiba-ai
python tools/daily_predict.py >> logs\daily_predict.log 2>&1
