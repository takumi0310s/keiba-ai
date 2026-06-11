@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if "%~1"=="" (
  echo Usage: realtime_5_9_runner.bat ^<subcmd^> [args]
  exit /b 1
)
python tools/realtime_5_9.py %* >> logs\realtime_5_9.log 2>&1
