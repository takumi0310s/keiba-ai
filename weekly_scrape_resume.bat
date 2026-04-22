@echo off
REM 毎週月曜 AM 06:30 - SCRAPER-GUARD解除直後にscrape_missing_all自動再開
REM 金曜22:00〜月曜06:00の停止中に貯まった欠落データを補填する

cd /d C:\Users\takum\keiba-ai

set LOGFILE=logs\scrape_missing_%date:~0,4%%date:~5,2%%date:~8,2%.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Weekly Scrape Resume Start >> %LOGFILE%

python -u tools\scrape_missing_all.py >> %LOGFILE% 2>&1

echo [%date% %time%] Weekly Scrape Resume End (exit=%ERRORLEVEL%) >> %LOGFILE%
