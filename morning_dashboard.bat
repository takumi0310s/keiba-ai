@echo off
cd /d C:\Users\takum\keiba-ai
set LOGFILE=logs\morning_dashboard_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo [%date% %time%] Morning Dashboard Start >> %LOGFILE%
python tools\morning_dashboard.py >> %LOGFILE% 2>&1
echo [%date% %time%] Morning Dashboard End (rc=%ERRORLEVEL%) >> %LOGFILE%
