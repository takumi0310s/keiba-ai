@echo off
REM process_watchdog 起動用バッチ (タスクスケジューラ登録用)
REM 継続監視モード。タスクスケジューラで "起動時" または 5分間隔で実行。
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\Users\takum\keiba-ai
python -u tools\process_watchdog.py --once
endlocal
