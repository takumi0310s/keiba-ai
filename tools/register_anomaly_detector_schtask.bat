@echo off
:: ============================================================
:: Sub-task T6: anomaly auto detector schtask 登録 script
:: ★ 5/18 user 判断後 admin 権限で 実行 ★
:: agent 内では 絶対に実行しない (本 file 作成のみ)
:: ============================================================
:: 5/17 G1 day 朝の各 chain timing で auto fire:
::   06:30  - DailyJrdbKyi 後の check
::   08:30  - DailyPredict 後の check
::   09:40  - 9:30 Discord 通知後の check (critical)
::   14:10  - 14:00 投票準備直前 check
::   17:00  - G1 後 evening 整理 check
:: ============================================================

set TR="C:\Users\takum\keiba-ai\tools\anomaly_auto_detector.bat"

schtasks /Create /TN "Keiba-AnomalyCheck-0630" /TR %TR% /SC DAILY /ST 06:30 /F
schtasks /Create /TN "Keiba-AnomalyCheck-0830" /TR %TR% /SC DAILY /ST 08:30 /F
schtasks /Create /TN "Keiba-AnomalyCheck-0940" /TR %TR% /SC DAILY /ST 09:40 /F
schtasks /Create /TN "Keiba-AnomalyCheck-1410" /TR %TR% /SC DAILY /ST 14:10 /F
schtasks /Create /TN "Keiba-AnomalyCheck-1700" /TR %TR% /SC DAILY /ST 17:00 /F

echo.
echo Done. 5 tasks registered. Confirm with:
echo   schtasks /Query /TN Keiba-AnomalyCheck-0630
echo.
