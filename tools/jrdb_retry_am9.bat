@echo off
chcp 65001 >nul
setlocal

REM JRDB AM 9:00 retry — 06:00 fetch で 404 だった (TYB/SED/KYI 等) を retry
REM 9:00 なら JRDB 公開済みのことが多い (publish タイミング 問題対策)

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\jrdb_retry_am9_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo [%date% %time%] JRDB AM 9:00 retry START (TODAY=%TODAY%) >> %LOGFILE%

REM 06:00 の DailyJrdbKyi で取得失敗した可能性のある type を retry。
REM publish タイミング: TYB は当日朝、SED は前日結果ベースで朝公開。
REM --force で既存 ZIP があっても再 DL、最新版上書き。

%PYTHON_EXE% tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1

REM Discord 通知 (取得結果)
%PYTHON_EXE% tools\jrdb_health_check.py --silent >> %LOGFILE% 2>&1

set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] JRDB AM 9:00 retry END exitcode=%EXITCODE% >> %LOGFILE%
endlocal
exit /b %EXITCODE%
