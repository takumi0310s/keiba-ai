@echo off
chcp 65001 >nul
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
set LOGFILE=logs\per_race_coverage.bat_%TODAY%.log
echo [%date% %time%] per_race_coverage.bat START >> %LOGFILE%
python tools\per_race_coverage_check.py --date %TODAY% >> %LOGFILE% 2>&1
echo [%date% %time%] per_race_coverage.bat END >> %LOGFILE%
exit /b 0
