@echo off
cd /d C:\Users\takum\keiba-ai
python tools/daily_predict.py >> logs\daily_predict.log 2>&1
