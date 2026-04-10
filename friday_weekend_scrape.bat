@echo off
REM 毎週金曜 AM 10:00 - 週末限定データ取得（波乱度・AI予測・データ分析）
REM ※ daily_premium_scrape.pyから自動呼出しされるが、単体実行も可能

cd /d C:\Users\takum\keiba-ai

REM Log file
set LOGFILE=logs\weekend_thisweek_%date:~0,4%%date:~5,2%%date:~8,2%.log

echo [%date% %time%] Weekend Thisweek Scrape Start >> %LOGFILE%

python tools\scrape_weekend_thisweek.py >> %LOGFILE% 2>&1

echo [%date% %time%] Weekend Thisweek Scrape End >> %LOGFILE%
