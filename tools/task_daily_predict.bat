@echo off
chcp 65001 >nul
setlocal

REM タスクスケジューラ用 daily_predict ランチャー（既存 daily_predict.bat と並存、手動切替可能）。
REM 特徴:
REM   - START /B で子プロセス化（セッションCLOSE耐性）
REM   - Intel Fortran / OpenMP の window-CLOSE クラッシュ対策 env
REM   - ログは logs\daily_predict_task_YYYYMMDD.log
REM   - 既存 daily_predict.bat は「絶対に変更しない」ポリシーのためこちらを新設

cd /d C:\Users\takum\keiba-ai

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\daily_predict_task_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
set KEIBA_OPERATIONAL_MODE=1

echo [%date% %time%] task_daily_predict Start (date=%TODAY%) >> %LOGFILE%

REM --resume を付けておくことで前回未完了時も自動再開できる
START "DailyPredictTask" /B python -u tools\daily_predict.py --date %TODAY% --resume >> %LOGFILE% 2>&1

echo [%date% %time%] task_daily_predict Launched (background) >> %LOGFILE%
exit /b 0
