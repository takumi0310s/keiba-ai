@echo off
chcp 65001 >nul
setlocal

REM daily_predict を START /B で子プロセス化して起動。
REM 親コンソールが閉じても生存しやすく、Intel Fortran の window-CLOSE 強制終了を回避する補助策。
REM Usage: tools\run_daily_predict.bat [YYYYMMDD]

cd /d C:\Users\takum\keiba-ai

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE

set TARGET_DATE=%1

if "%TARGET_DATE%"=="" (
    START "DailyPredict" /B python -u tools\daily_predict.py
) else (
    START "DailyPredict" /B python -u tools\daily_predict.py --date %TARGET_DATE%
)
exit /b 0
