@echo off
chcp 65001 >nul
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
set LOGFILE=logs\paper_s2b_results.bat_%TODAY%.log
echo [%date% %time%] paper_s2b_results.bat START >> %LOGFILE%
python tools\paper_trade_s2b.py results --date %TODAY% >> %LOGFILE% 2>&1
echo [%date% %time%] paper_s2b_results.bat END >> %LOGFILE%
exit /b 0
