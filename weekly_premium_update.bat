@echo off
cd /d %~dp0
echo [%date% %time%] Weekly Premium Update starting...
python tools\weekly_premium_update.py >> logs\weekly_premium_update.log 2>&1
echo [%date% %time%] Weekly Premium Update done. >> logs\weekly_premium_update.log
