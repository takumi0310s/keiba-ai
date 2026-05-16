@echo off
REM ★ 5/18 user 判断後、 admin 権限で実行 ★
REM 毎日 21:00 fire (DailyResultsEvening 20:00 の 1 時間後)
REM agent 内では絶対に実行しない (sub-task 10 制約)

schtasks /Create /TN "Keiba-DailyCumulativeAudit" ^
  /TR "C:\Users\takum\keiba-ai\tools\daily_cumulative_audit.bat" ^
  /SC DAILY /ST 21:00 ^
  /RL HIGHEST /F

echo.
echo Keiba-DailyCumulativeAudit 登録完了 (毎日 21:00 fire)
echo 動作確認: schtasks /Run /TN "Keiba-DailyCumulativeAudit"
echo logs: logs\daily_cumulative_audit.log
