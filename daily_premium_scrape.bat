@echo off
REM 毎日AM3:00にタスクスケジューラで自動実行
REM プレミアムデータの分散取得（500レース/ソース/日）

cd /d C:\Users\sato\keiba-ai

REM ログファイル
set LOGFILE=logs\premium_scrape_%date:~0,4%%date:~5,2%%date:~8,2%.log

echo [%date% %time%] Daily Premium Scrape Start >> %LOGFILE%

python tools\daily_premium_scrape.py --limit 500 >> %LOGFILE% 2>&1

echo [%date% %time%] Daily Premium Scrape End >> %LOGFILE%
