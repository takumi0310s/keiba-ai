@echo off
rem [2026-08-11 供給復旧] netkeiba解約に伴い無効化 (週末premium+旧Paci週次)。
rem Paci供給は keiba-ai\JrdbSupplyDaily (tools/daily_jrdb_supply.bat) が日次で代替。
rem 原本= friday_weekend_scrape.bat.bak_20260811。タスク disable は要管理者:
rem   schtasks /change /tn "Keiba-FridayWeekendScrape" /disable
exit /b 0
