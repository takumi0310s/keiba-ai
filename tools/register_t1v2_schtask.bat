@echo off
rem T1v2 監査タスク登録 (管理者権限で実行)
schtasks /create /tn "KeibaAI_T1v2_Audit" /tr "C:\Users\takum\keiba-ai\tools\t1v2_audit.bat" /sc daily /st 08:50 /f
if %errorlevel%==0 (echo T1v2 task registered) else (echo FAILED - run as admin)
