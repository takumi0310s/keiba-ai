@echo off
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I
rem Œn“C: •½“ú[–é ·•ª’Ç (JRDB+JV“‡E2026-08-17`B‹ŒJrdbSupplyDaily+JvlinkSupplyDaily‚ÌŒãŒp)
python tools\daily_jrdb_supply.py >> logs\daily_jrdb_supply_%TODAY%.log 2>&1
python tools\daily_jvlink_supply.py >> logs\daily_jvlink_supply_%TODAY%.log 2>&1
