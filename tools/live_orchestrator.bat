@echo off
:: P0-5 Live Orchestrator (★ 5/18+ admin schtask から fire ★)
:: SAT/SUN 08:30 想定、 internal polling loop で 各 race -15 min 処理

setlocal

cd /d C:\Users\takum\keiba-ai
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe

:: log dir 確保 (初回 fire 時の stdout.log 書き込み対策)
if not exist data\live_orchestrator_log mkdir data\live_orchestrator_log

:: ★ 5/24 (SAT) 以降は mock 解除予定、 5/17-5/23 は mock=True 強制 ★
:: 今は mock=True で安全運用、 5/24 以降 user 判断で mock 解除

%PYTHON_EXE% -u tools\live_orchestrator_main.py --mock --dry-run >> data\live_orchestrator_log\stdout.log 2>&1

endlocal
