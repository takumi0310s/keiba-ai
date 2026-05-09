@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if "%~1"=="" (
  python tools/stage2_predict.py --check-next-1h >> logs\stage2_predict.log 2>&1
) else (
  python tools/stage2_predict.py %* >> logs\stage2_predict.log 2>&1
)
