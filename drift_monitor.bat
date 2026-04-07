@echo off
cd /d C:\Users\takum\keiba-ai
python tools/drift_monitor.py --verbose >> logs\drift_monitor.log 2>&1
