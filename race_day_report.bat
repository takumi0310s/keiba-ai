@echo off
rem 2026-08-15 取捨確定: RaceDayReport は 4月から恒常不発 (daily_results と同刻起動で毎回「該当なし」) かつ
rem daily_results 18:00 のリッチ通知と役割重複のため停止。タスク disable は Access denied のため bat スタブ化。
rem 正式 disable: tools\disable_legacy_audit_admin.bat を管理者実行。復元: copy race_day_report.bat.bak_20260815 race_day_report.bat
exit /b 0
