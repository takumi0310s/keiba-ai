@echo off
REM Keiba-NightlySanity - Daily PM23:00 pre-check for next day's auto-fired tasks

cd /d C:\Users\takum\keiba-ai

set LOGFILE=logs\nightly_sanity_%date:~0,4%%date:~5,2%%date:~8,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Nightly Sanity Check Start >> %LOGFILE%
python tools\nightly_sanity_check.py >> %LOGFILE% 2>&1
echo [%date% %time%] Nightly Sanity Check End (rc=%ERRORLEVEL%) >> %LOGFILE%
