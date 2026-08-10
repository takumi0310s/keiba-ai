@echo off
rem JRDB 日次供給 (2026-08-11 供給復旧)。旧 DailyJrdbKyi/金曜Paci チェーンの代替
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
python tools\daily_jrdb_supply.py >> logs\daily_jrdb_supply_%TODAY%.log 2>&1
