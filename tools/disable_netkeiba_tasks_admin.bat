@echo off
rem 管理者権限で実行: netkeiba premium 系タスクの正式 disable
schtasks /change /tn "keiba-ai\DailyPremiumScrape" /disable
schtasks /change /tn "Keiba-FridayWeekendScrape" /disable
schtasks /change /tn "keiba-ai\RaceAutoNotify_Sat" /disable
schtasks /change /tn "keiba-ai\RaceAutoNotify_Sun" /disable
echo done
