@echo off
:: Sub-task T6: anomaly auto detector (Discord 通知 + log)
:: 5/17 G1 day で 5 trigger 自動 check、 user 操作 = Discord 確認のみ
cd /d "C:\Users\takum\keiba-ai"
if not exist logs mkdir logs
python tools\anomaly_auto_detector.py >> logs\anomaly_auto_detector.log 2>&1
exit /b %ERRORLEVEL%
