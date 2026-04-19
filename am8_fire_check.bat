@echo off
cd /d C:\Users\takum\keiba-ai
set LOGFILE=logs\am8_fire_check_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo [%date% %time%] AM8 Fire Check Start >> %LOGFILE%
python tools\am8_fire_check.py >> %LOGFILE% 2>&1
echo [%date% %time%] AM8 Fire Check End (rc=%ERRORLEVEL%) >> %LOGFILE%
