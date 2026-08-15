@echo off
rem Phase1 �K�p (2026-08-17): �n��C���� + �č��n�������Bdenied �� bat �X�^�u�őΏ�
cd /d C:\Users\takum\keiba-ai
set LOG=logs\phase1_apply.log
echo [%date% %time%] phase1 apply start >> %LOG%
schtasks /change /tn "keiba-ai\JrdbSupplyDaily" /disable >> %LOG% 2>&1
schtasks /change /tn "keiba-ai\JvlinkSupplyDaily" /disable >> %LOG% 2>&1
rem 8/15 取捨確定で AM3/AM6 は disable 済み (二重適用防止のため削除)
schtasks /change /tn "Keiba-AM8FireCheck" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-MorningDigest" /disable >> %LOG% 2>&1
schtasks /change /tn "keiba-ai\DataFreshnessMonitor" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-TybPublishMonitor" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-JrdbRetryAm9_Sat" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-JrdbRetryAm9_Sun" /disable >> %LOG% 2>&1
echo [%date% %time%] phase1 apply done (denied �� disable_legacy_audit_admin.bat �ŊǗ��Ҏ��s) >> %LOG%
