@echo off
rem T1v2 ŠÄ¸: “y“ú=ƒ‰ƒCƒudumpŠÄ¸ / •½“ú=‹Ÿ‹‹source-check
cd /d C:\Users\takum\keiba-ai
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Date).DayOfWeek.value__"`) do set DOW=%%I
if %DOW%==6 goto dump
if %DOW%==0 goto dump
python tools\t1v2_feature_audit.py --source-check >> logs\t1v2_audit.log 2>&1
goto end
:dump
python tools\t1v2_feature_audit.py >> logs\t1v2_audit.log 2>&1
:end
