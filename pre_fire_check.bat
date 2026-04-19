@echo off
REM Keiba-PreFireCheck — AM02:55 daily pre-fire preventive check

cd /d C:\Users\takum\keiba-ai

set LOGFILE=logs\pre_fire_check_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Pre-Fire-Check Start >> %LOGFILE%
python tools\pre_fire_check.py >> %LOGFILE% 2>&1
echo [%date% %time%] Pre-Fire-Check End (rc=%ERRORLEVEL%) >> %LOGFILE%
