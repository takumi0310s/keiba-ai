@echo off
cd /d C:\Users\takum\keiba-ai
set LOG_DATE=%date:~0,4%%date:~5,2%%date:~8,2%
python tools\strategy8_sidecar.py --auto >> logs\strategy8_sidecar_%LOG_DATE%.log 2>&1
