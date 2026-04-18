@echo off
REM Sat/Sun AM 07:30 - JRDB data health check (insurance for DailyPremiumScrape)

cd /d C:\Users\takum\keiba-ai

set LOGFILE=logs\jrdb_health_check_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] JRDB Health Check Start >> %LOGFILE%
python tools\jrdb_health_check.py >> %LOGFILE% 2>&1
echo [%date% %time%] JRDB Health Check End (rc=%ERRORLEVEL%) >> %LOGFILE%
