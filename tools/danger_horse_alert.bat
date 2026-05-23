@echo off
chcp 65001 >nul
setlocal
cd /d c:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\danger_horse_alert_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo [%date% %time%] DangerHorseAlert start (TODAY=%TODAY%) >> %LOGFILE%

%PYTHON_EXE% tools\danger_horse_alert.py --date %TODAY% --discord >> %LOGFILE% 2>&1

echo [%date% %time%] DangerHorseAlert done >> %LOGFILE%
