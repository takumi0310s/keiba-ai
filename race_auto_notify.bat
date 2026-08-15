@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
rem [2026-08-11 通知再整備 item4] paper期間中は5分前通知ループを停止 (提案実装)。
rem 復元 = copy race_auto_notify.bat.bak_20260811 race_auto_notify.bat
rem 09:30 朝一括通知 = keiba-ai\MorningBatchNotify (tools/morning_batch_notify.bat)
rem タスク本体の disable は要管理者 (tools/disable_netkeiba_tasks_admin.bat に追記済)
exit /b 0
