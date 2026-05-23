@echo off
REM TYB publish タイミング 観測 (毎時実行)
REM Usage: tyb_publish_monitor.bat [YYYYMMDD]

setlocal
chcp 65001 >nul
cd /d C:\Users\takum\keiba-ai

set PYTHONIOENCODING=utf-8
set PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

REM 5/4-5/10 の期間中、weekend (Sat=20260509, Sun=20260510) を観測対象に
REM デフォルト: 今日の race day を観測
REM 引数で日付指定可

if "%~1"=="" (
    REM 今日の曜日を確認、平日は次の Sat/Sun を target
    for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$d = Get-Date; if ($d.DayOfWeek -in 'Saturday','Sunday') { $d.ToString('yyyyMMdd') } else { (0..6 | ForEach-Object { $d.AddDays($_) } | Where-Object { $_.DayOfWeek -eq 'Saturday' } | Select-Object -First 1).ToString('yyyyMMdd') }"`) do set TARGET=%%D
) else (
    set TARGET=%~1
)

set LOGFILE=logs\tyb_publish_monitor_%TODAY%.log

echo [%date% %time%] TYB monitor TARGET=%TARGET% >> %LOGFILE%
%PYTHON_EXE% tools\tyb_publish_monitor.py --date %TARGET% >> %LOGFILE% 2>&1
echo [%date% %time%] TYB monitor end >> %LOGFILE%

exit /b 0
