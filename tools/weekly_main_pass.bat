@echo off
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
rem Œn“A: ‹à—j–é TŽŸƒƒCƒ“ƒpƒX (2026-08-21`)BT––ƒoƒ“ƒhƒ‹+JV¡TŒn‚ð‘O–é‚ÉŽæ“¾
python tools\daily_jrdb_supply.py >> logs\weekly_main_pass_%TODAY%.log 2>&1
python tools\daily_jvlink_supply.py >> logs\weekly_main_pass_%TODAY%.log 2>&1
C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\jv_daily_fetch.ps1 -specs "TCOV,RCOV" -openOption 2 >> logs\weekly_main_pass_%TODAY%.log 2>&1
python tools\jv_daily_parse.py >> logs\weekly_main_pass_%TODAY%.log 2>&1
