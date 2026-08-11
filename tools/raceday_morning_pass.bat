@echo off
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
rem Œn“B: ŠJÃ“ú’© “–“úƒpƒX (2026-08-22`AŽb’è06:50=8/15probeŒ‹‰Ê+15•ª‚ÅŠm’è)
rem “–“úTYB/KAB·•ª + JV·•ª‚Ì‚Ý (d‚¢TŽŸ‚Í‹à—j–éA‚ÅÏ‚Ý)
python tools\daily_jrdb_supply.py >> logs\raceday_morning_%TODAY%.log 2>&1
python tools\daily_jvlink_supply.py >> logs\raceday_morning_%TODAY%.log 2>&1
