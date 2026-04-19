@echo off
REM Keiba-AM3FireCheck — AM3:15 daily reverse-watchdog for DailyPremiumScrape

cd /d C:\Users\takum\keiba-ai

set LOGFILE=logs\am3_fire_check_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] AM3 Fire Check Start >> %LOGFILE%
python tools\am3_fire_check.py >> %LOGFILE% 2>&1
echo [%date% %time%] AM3 Fire Check End (rc=%ERRORLEVEL%) >> %LOGFILE%
