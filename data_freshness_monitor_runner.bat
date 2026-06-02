@echo off
REM データ鮮度監視。 毎朝 07:30 silent起動。 主要データの当日カバレッジを Discord #アップデート に警告。
REM ★ 読み取り + 通知のみ。 予測ロジック不変 ★
REM 2026-05-31 修正: silent_runner下でハングする for/f powershell を撤廃、固定名ログに変更。
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo [START] data_freshness_monitor >> logs\data_freshness_monitor.log 2>&1
C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe tools\data_freshness_monitor.py >> logs\data_freshness_monitor.log 2>&1
echo [END exit=%ERRORLEVEL%] data_freshness_monitor >> logs\data_freshness_monitor.log 2>&1
