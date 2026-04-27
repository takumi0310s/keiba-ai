@echo off
REM Race auto-predict and Discord notify (5 min before each race)
REM Runs Sat/Sun AM 09:30 via Task Scheduler

cd /d C:\Users\takum\keiba-ai

REM Prevent PC sleep during execution
powercfg /change standby-timeout-ac 0 2>nul

set LOGFILE=logs\race_auto_notify_%date:~0,4%%date:~5,2%%date:~8,2%.log
echo [%date% %time%] Race Auto-Notify Start >> %LOGFILE%

REM Unbuffered output + UTF-8 to prevent encoding crashes
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

python tools\race_auto_notify.py >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo [%date% %time%] Race Auto-Notify End (exit=%EXITCODE%) >> %LOGFILE%

REM Restore sleep timeout (default 30 min)
powercfg /change standby-timeout-ac 30 2>nul
exit /B %EXITCODE%
