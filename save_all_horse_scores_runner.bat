@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python tools/save_all_horse_scores.py %* >> logs\save_all_horse_scores.log 2>&1
