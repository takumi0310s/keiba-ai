@echo off
rem JV-Link “úŽŸ‹Ÿ‹‹ (2026-08-12)
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
python tools\daily_jvlink_supply.py >> logs\daily_jvlink_supply_%TODAY%.log 2>&1
