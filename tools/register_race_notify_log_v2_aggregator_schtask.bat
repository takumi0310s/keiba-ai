@echo off
REM ★ 5/18 admin 登録専用 (管理者権限で実行) ★
REM
REM race_notify_log v2 aggregator — DAILY 20:30
REM DailyResultsEvening (20:00) 完了後に 3 phase log を集計
REM
REM Output: data/race_notify_log_v2_summary/summary_YYYYMMDD_HHMMSS.json
REM
REM ★ 実 schtasks /create はこの .bat を admin で手動実行する必要あり ★
REM ★ 本 sub-task では実 registration は行わない、 5/18 admin tasks に追加のみ ★

setlocal
set REPO=%~dp0..
set PYTHON_EXE=python

schtasks /Create ^
  /TN "Keiba-RaceNotifyLogV2-Aggregator" ^
  /TR "cmd /c cd /d %REPO% && %PYTHON_EXE% tools\race_notify_log_v2_aggregator.py >> logs\race_notify_log_v2_aggregator.log 2>&1" ^
  /SC DAILY ^
  /ST 20:30 ^
  /RL HIGHEST ^
  /F

if %ERRORLEVEL% EQU 0 (
    echo OK: Keiba-RaceNotifyLogV2-Aggregator registered (DAILY 20:30)
) else (
    echo ERROR: registration failed - need admin?
)

endlocal
