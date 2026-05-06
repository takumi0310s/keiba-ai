@echo off
chcp 65001 >nul
setlocal

REM 当日 3 段階予測機構 wrapper
REM Usage: multi_stage_predict.bat <stage> [YYYYMMDD]
REM   stage: test10 | race11_1450 | race12_1545

if "%~1"=="" (
    echo [NG] stage 引数 必須: test10 / race11_1450 / race12_1545
    exit /b 2
)
set STAGE=%~1

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\multi_stage_predict_%STAGE%_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] multi_stage_predict.bat START stage=%STAGE% >> %LOGFILE%

if "%~2"=="" (
    python -u tools\multi_stage_predict.py --stage %STAGE% >> %LOGFILE% 2>&1
) else (
    python -u tools\multi_stage_predict.py --stage %STAGE% --date %~2 >> %LOGFILE% 2>&1
)

set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] multi_stage_predict.bat END exitcode=%EXITCODE% >> %LOGFILE%
endlocal
exit /b %EXITCODE%
