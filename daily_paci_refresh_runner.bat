@echo off
REM PACI 当日取得 + 安全再生成。 Task Scheduler から silent_runner.vbs 経由で土日朝 06:50 起動。
REM ★ V15 model / predict_core / daily_predict 不変。 paciデータ取得経路の修復のみ ★
REM 2026-05-31 修正: silent_runner下でハングする for/f powershell を撤廃、固定名ログに変更
REM   (python側が冒頭でISO時刻を出力するので日付は識別可能)。
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo [START] daily_paci_refresh >> logs\daily_paci_refresh.log 2>&1
C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe tools\daily_paci_refresh.py >> logs\daily_paci_refresh.log 2>&1
echo [END exit=%ERRORLEVEL%] daily_paci_refresh >> logs\daily_paci_refresh.log 2>&1
