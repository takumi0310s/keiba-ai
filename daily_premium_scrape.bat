@echo off
REM Daily AM 03:00 - Premium data pre-fetch (500 races/source/day)

cd /d C:\Users\takum\keiba-ai

REM Log file
set LOGFILE=logs\premium_scrape_%date:~0,4%%date:~5,2%%date:~8,2%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Daily Premium Scrape Start >> %LOGFILE%

python tools\daily_premium_scrape.py >> %LOGFILE% 2>&1

echo [%date% %time%] Daily Premium Scrape End >> %LOGFILE%
