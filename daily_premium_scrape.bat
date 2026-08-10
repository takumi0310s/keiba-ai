@echo off
rem [2026-08-11 供給復旧] netkeiba解約に伴い無効化 (premium一括スクレイプ)。
rem 原本= daily_premium_scrape.bat.bak_20260811。タスク本体の disable は要管理者:
rem   schtasks /change /tn "keiba-ai\DailyPremiumScrape" /disable
exit /b 0
