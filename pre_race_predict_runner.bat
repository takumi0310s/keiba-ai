@echo off
REM Session #77: silent_runner.vbs Line 24 ERROR_FILE_NOT_FOUND fix
REM stage2_predict.py is on dev/two-stage only, not on main.
REM On main, no-op exit 0 to prevent Windows Script Host popup.
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist logs mkdir logs
echo [%DATE% %TIME%] pre_race_predict_runner stub args=%* >> logs\pre_race_predict.log
if exist tools\stage2_predict.py (
  python tools\stage2_predict.py %* >> logs\pre_race_predict.log 2>&1
) else (
  echo [%DATE% %TIME%]   stage2_predict.py not on main, no-op exit 0 >> logs\pre_race_predict.log
)
exit /b 0
