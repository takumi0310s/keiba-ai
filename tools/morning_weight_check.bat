@echo off
chcp 65001 >nul
setlocal

REM 09:30 (土) 起動用 タスクスケジューラ wrapper
REM Usage: morning_weight_check.bat [YYYYMMDD]

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\morning_weight_check_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] morning_weight_check.bat START >> %LOGFILE%

if "%~1"=="" (
    python -u tools\morning_weight_check.py >> %LOGFILE% 2>&1
) else (
    python -u tools\morning_weight_check.py --date %~1 >> %LOGFILE% 2>&1
)

set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] morning_weight_check.bat END exitcode=%EXITCODE% >> %LOGFILE%
endlocal
exit /b %EXITCODE%
