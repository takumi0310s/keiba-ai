@echo off
:: ============================================================
:: Keiba-FeaturesIntegrityCheck (DAILY 22:00)
:: ★ 完成-1 sub-task (5/18) で作成 ★
:: schtask 登録は admin 別 sub-task (本 bat は schtasks /Create 含まず)
::
:: 実 module: tools/features_integrity_monitor.py
:: 注記: 仕様書 T1_features_integrity_check.py は未存在、
::       実 module は features_integrity_monitor.py
:: ============================================================
cd /d C:\Users\takum\keiba-ai
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
%PYTHON_EXE% -u tools\features_integrity_monitor.py >> logs\keiba_features_integrity_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
