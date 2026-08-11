@echo off
rem ŠÇ—ÒŒ ŒÀ‚ÅÀs: Phase1 ‚Ì denied •ª‚ğ³® disable
schtasks /change /tn "Keiba-AM3FireCheck" /disable
schtasks /change /tn "Keiba-AM6FireCheck" /disable
schtasks /change /tn "Keiba-AM8FireCheck" /disable
schtasks /change /tn "Keiba-MorningDigest" /disable
schtasks /change /tn "Keiba-TybPublishMonitor" /disable
schtasks /change /tn "Keiba-JrdbRetryAm9_Sat" /disable
schtasks /change /tn "Keiba-JrdbRetryAm9_Sun" /disable
echo done
