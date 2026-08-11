@echo off
rem Phase1 適用 (2026-08-17): 系統C統合 + 監査系無効化。denied は bat スタブで対処
cd /d C:\Users\takum\keiba-ai
set LOG=logs\phase1_apply.log
echo [%date% %time%] phase1 apply start >> %LOG%
schtasks /change /tn "keiba-ai\JrdbSupplyDaily" /disable >> %LOG% 2>&1
schtasks /change /tn "keiba-ai\JvlinkSupplyDaily" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-AM3FireCheck" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-AM6FireCheck" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-AM8FireCheck" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-MorningDigest" /disable >> %LOG% 2>&1
schtasks /change /tn "keiba-ai\DataFreshnessMonitor" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-TybPublishMonitor" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-JrdbRetryAm9_Sat" /disable >> %LOG% 2>&1
schtasks /change /tn "Keiba-JrdbRetryAm9_Sun" /disable >> %LOG% 2>&1
echo [%date% %time%] phase1 apply done (denied は disable_legacy_audit_admin.bat で管理者実行) >> %LOG%
