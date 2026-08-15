@echo off
rem �Ǘ��Ҍ����Ŏ��s: Phase1 �� denied ���𐳎� disable
rem 8/15 取捨確定: AM3/AM6 は disable 済みのため削除
schtasks /change /tn "Keiba-AM8FireCheck" /disable
rem 8/15 取捨確定: RaceDayReport は非管理者で Access denied → bat スタブ化済み。ここで正式 disable
schtasks /change /tn "Keiba-RaceDayReport_Sat" /disable
schtasks /change /tn "Keiba-RaceDayReport_Sun" /disable
schtasks /change /tn "Keiba-MorningDigest" /disable
schtasks /change /tn "Keiba-TybPublishMonitor" /disable
schtasks /change /tn "Keiba-JrdbRetryAm9_Sat" /disable
schtasks /change /tn "Keiba-JrdbRetryAm9_Sun" /disable
echo done
