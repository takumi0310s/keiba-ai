@echo off
chcp 65001 > nul
cd /d C:\Users\takum\keiba-ai
call .venv\Scripts\activate.bat
python tools/drift_detector.py >> logs\drift_detector_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
