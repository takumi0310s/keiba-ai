@echo off
REM nar_daily_pipeline.bat — NAR 当日 pipeline (scrape → predict)
REM Usage: nar_daily_pipeline.bat [YYYYMMDD]
REM   引数なし: 今日
REM タスクスケジューラからは silent_runner.vbs 経由で起動推奨 (静音化)

setlocal enabledelayedexpansion

set BASE=C:\Users\takum\keiba-ai
cd /d "%BASE%"

REM 引数 → DATE
if "%~1"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set DT=%%I
    if not defined DT (
        REM Windows 11 24H2 wmic 不在 fallback
        for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%I
    )
    set DATE_ARG=!DT:~0,8!
) else (
    set DATE_ARG=%~1
)

set LOG_DIR=%BASE%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG=%LOG_DIR%\nar_daily_!DATE_ARG!.log

echo [%DATE% %TIME%] === NAR daily pipeline START (date=!DATE_ARG!) === >> "%LOG%" 2>&1

REM SCRAPER-GUARD は既存 (tools/scraper_guard.py)。NAR scrape は別チャネル想定なので明示 skip 可能
REM 必要なら下記 1 行を有効化:
REM python tools\scraper_guard.py --caller nar_daily_pipeline --mode exit >> "%LOG%" 2>&1
REM if errorlevel 1 ( echo SCRAPER-GUARD blocked >> "%LOG%" & exit /b 0 )

REM 1. 当日 race 出馬表 + odds 取得 (script は将来追加、今は predict のみ)
REM python tools\scrape_nar_today.py --date !DATE_ARG! >> "%LOG%" 2>&1
REM if errorlevel 1 ( echo NAR scrape failed >> "%LOG%" & exit /b 1 )

REM 2. predict 実行
python tools\predict_nar.py --date !DATE_ARG! --output-csv data\daily_predictions\nar_!DATE_ARG!.csv >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] predict_nar failed >> "%LOG%"
    python tools\notify_done.py "NAR predict 失敗 !DATE_ARG!" "tools\predict_nar.py がエラー終了。logs\nar_daily_!DATE_ARG!.log を確認。" --color red >> "%LOG%" 2>&1
    exit /b 1
)

echo [%DATE% %TIME%] === NAR daily pipeline END === >> "%LOG%"
endlocal
