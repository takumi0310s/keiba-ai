@echo off
REM 毎日 AM 07:00 - scrape_missing の進捗確認 (軽量)

cd /d C:\Users\takum\keiba-ai

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

python -u tools\scrape_progress_check.py >> logs\scrape_progress.log 2>&1
