@echo off
REM 朝 06:30 起動用 タスクスケジューラ wrapper
REM Usage: morning_top_races.bat [YYYYMMDD]
REM
REM 引数なしで実行 → 今日の日付で morning_top_races.sh を呼ぶ
REM Git Bash 経由で .sh を実行

setlocal
chcp 65001 >nul

cd /d C:\Users\takum\keiba-ai

REM Git Bash パス検出
set BASH_EXE=C:\Program Files\Git\bin\bash.exe
if not exist "%BASH_EXE%" set BASH_EXE=C:\Program Files (x86)\Git\bin\bash.exe
if not exist "%BASH_EXE%" (
    echo [ERROR] Git Bash not found
    exit /b 1
)

REM ログ用日付
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\morning_top_races_wrapper_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] morning_top_races.bat START (TODAY=%TODAY%) >> %LOGFILE%

REM 引数渡し: 引数指定なら $1、なければ今日
if "%~1"=="" (
    "%BASH_EXE%" -c "cd /c/Users/takum/keiba-ai && bash tools/morning_top_races.sh" >> %LOGFILE% 2>&1
) else (
    "%BASH_EXE%" -c "cd /c/Users/takum/keiba-ai && bash tools/morning_top_races.sh %~1" >> %LOGFILE% 2>&1
)

set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] morning_top_races.bat END exitcode=%EXITCODE% >> %LOGFILE%

exit /b %EXITCODE%
