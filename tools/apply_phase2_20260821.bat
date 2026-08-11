@echo off
rem Phase2 “K—p (2026-08-21‹à –é): Œn“B/E Š®‘S”Å‚ÖØ‘Ö
cd /d C:\Users\takum\keiba-ai
set LOG=logs\phase2_apply.log
echo [%date% %time%] phase2 apply start >> %LOG%
schtasks /change /tn "keiba-ai\MorningBatchNotify" /tr "C:\Users\takum\keiba-ai\tools\morning_batch_notify_v2.bat" >> %LOG% 2>&1
schtasks /change /tn "keiba-ai\JrdbSupplyWeekendAM" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-MorningWeightCheck_Sat" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-MorningWeightCheck_Sun" /disable >> %LOG% 2>&1
echo [%date% %time%] phase2 apply done >> %LOG%
