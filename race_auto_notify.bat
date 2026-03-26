@echo off
REM レース5分前自動予測＆Discord通知
REM 土日AM9:30にタスクスケジューラで自動起動

cd /d C:\Users\sato\keiba-ai

REM PCスリープ防止
powercfg /change standby-timeout-ac 0

set LOGFILE=logs\race_auto_notify_%date:~0,4%%date:~5,2%%date:~8,2%.log
echo [%date% %time%] Race Auto-Notify Start >> %LOGFILE%

python tools\race_auto_notify.py >> %LOGFILE% 2>&1

echo [%date% %time%] Race Auto-Notify End >> %LOGFILE%

REM スリープ防止を解除（デフォルト30分に戻す）
powercfg /change standby-timeout-ac 30
