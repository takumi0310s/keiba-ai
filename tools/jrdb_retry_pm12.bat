@echo off
chcp 65001 >nul
setlocal

REM JRDB PM 12:00 final retry — 06:00 / 09:00 共に失敗時の最終 retry
REM 12:00 は当日朝公開後の確実な取得タイミング (publish 完了想定)
REM 失敗時 Discord 警告: 「JRDB なしで投資判断」 通知

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\jrdb_retry_pm12_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] JRDB PM 12:00 FINAL retry START (TODAY=%TODAY%) >> %LOGFILE%

REM 09:00 retry も失敗していた場合の最終手段
python tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1

REM 健全性 check + Discord 通知 (失敗あれば critical 通知)
python tools\jrdb_health_check.py >> %LOGFILE% 2>&1

set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] JRDB PM 12:00 FINAL retry END exitcode=%EXITCODE% >> %LOGFILE%

REM exit code != 0 (= JRDB 不完全) なら 「JRDB なしで投資判断」 を Discord に明示通知
if not %EXITCODE%==0 (
    python tools\notify_done.py "JRDB 12:00 FINAL retry も失敗" "5/9 投資は JRDB なしで判断。 V15 model は JRDB なしでも動作可能 (前走成績結合率は低下、約 75 -> 50 程度)。 案B改 V15 maintain、撤退ライン余裕で投資判断。 詳細: logs/jrdb_retry_pm12_%TODAY%.log" --color yellow >> %LOGFILE% 2>&1
)

endlocal
exit /b %EXITCODE%
