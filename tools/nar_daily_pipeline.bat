@echo off
REM nar_daily_pipeline.bat - NAR pipeline stage dispatcher
REM
REM Usage:
REM   nar_daily_pipeline.bat <stage> [YYYYMMDD]
REM
REM stages:
REM   scrape_today    16:30 - shutuba + estimated odds
REM   predict         17:00 - NAR v4 inference + candidates
REM   scrape_results  21:30 - results + payouts
REM   calendar        13:00 - scrape_nar_calendar (light list scrape)
REM   live_odds       19:00 - scrape_nar_live_odds (確定 odds refresh)
REM   all             one-shot test (scrape_today + predict)
REM
REM No arg or unknown stage falls back to "all"
REM scheduled tasks call this via silent_runner.vbs (hidden window)

setlocal enabledelayedexpansion

set BASE=C:\Users\takum\keiba-ai
cd /d "%BASE%"

set STAGE=%~1
if "%STAGE%"=="" set STAGE=all

if "%~2"=="" (
    for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DATE_ARG=%%I
) else (
    set DATE_ARG=%~2
)

set LOG_DIR=%BASE%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG=%LOG_DIR%\nar_daily_!STAGE!_!DATE_ARG!.log

echo [%DATE% %TIME%] === NAR pipeline stage=!STAGE! date=!DATE_ARG! START === >> "%LOG%" 2>&1

if /i "!STAGE!"=="scrape_today" goto stage_scrape_today
if /i "!STAGE!"=="predict" goto stage_predict
if /i "!STAGE!"=="scrape_results" goto stage_scrape_results
if /i "!STAGE!"=="calendar" goto stage_calendar
if /i "!STAGE!"=="live_odds" goto stage_live_odds
if /i "!STAGE!"=="all" goto stage_all
echo [WARN] unknown stage, fallback to all >> "%LOG%"
goto stage_all

:stage_scrape_today
echo [STAGE scrape_today] >> "%LOG%"
python tools\scrape_nar_today.py --date !DATE_ARG! >> "%LOG%" 2>&1
if errorlevel 1 (
    echo scrape_nar_today failed >> "%LOG%"
    python tools\notify_done.py "NAR scrape_today FAIL !DATE_ARG!" "scrape_nar_today error - check log" --color red >> "%LOG%" 2>&1
    exit /b 1
)
goto end

:stage_predict
echo [STAGE predict] >> "%LOG%"
python tools\predict_nar.py --date !DATE_ARG! --output-csv data\daily_predictions\nar_!DATE_ARG!.csv >> "%LOG%" 2>&1
if errorlevel 1 (
    echo predict_nar failed >> "%LOG%"
    python tools\notify_done.py "NAR predict FAIL !DATE_ARG!" "predict_nar error - check log" --color red >> "%LOG%" 2>&1
    exit /b 1
)
goto end

:stage_scrape_results
echo [STAGE scrape_results] >> "%LOG%"
python tools\scrape_nar_results.py --date !DATE_ARG! >> "%LOG%" 2>&1
if errorlevel 1 (
    echo scrape_nar_results failed >> "%LOG%"
    python tools\notify_done.py "NAR scrape_results FAIL !DATE_ARG!" "scrape_nar_results error - check log" --color red >> "%LOG%" 2>&1
    exit /b 1
)
goto end

:stage_calendar
echo [STAGE calendar] >> "%LOG%"
python tools\scrape_nar_calendar.py --date !DATE_ARG! >> "%LOG%" 2>&1
if errorlevel 1 (
    echo scrape_nar_calendar failed >> "%LOG%"
    python tools\notify_done.py "NAR calendar FAIL !DATE_ARG!" "scrape_nar_calendar error - check log" --color red >> "%LOG%" 2>&1
    exit /b 1
)
goto end

:stage_live_odds
echo [STAGE live_odds] >> "%LOG%"
python tools\scrape_nar_live_odds.py --date !DATE_ARG! --update-shutuba >> "%LOG%" 2>&1
if errorlevel 1 (
    echo scrape_nar_live_odds failed >> "%LOG%"
    python tools\notify_done.py "NAR live_odds FAIL !DATE_ARG!" "scrape_nar_live_odds error - check log" --color red >> "%LOG%" 2>&1
    exit /b 1
)
goto end

:stage_all
echo [STAGE all - test mode] >> "%LOG%"
python tools\scrape_nar_today.py --date !DATE_ARG! >> "%LOG%" 2>&1
if errorlevel 1 echo scrape_today failed (continuing) >> "%LOG%"
python tools\predict_nar.py --date !DATE_ARG! --output-csv data\daily_predictions\nar_!DATE_ARG!.csv >> "%LOG%" 2>&1
if errorlevel 1 echo predict failed >> "%LOG%"
goto end

:end
echo [%DATE% %TIME%] === NAR pipeline stage=!STAGE! END === >> "%LOG%"
endlocal
exit /b 0
