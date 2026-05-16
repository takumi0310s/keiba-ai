@echo off
REM ============================================================
REM Keiba-FeaturesIntegrityCheck schtask 登録 script
REM
REM ★ 5/18 user 判断後、 ADMIN 権限で実行 ★
REM ★ agent 内で実 schtasks /create 絶対実行しない ★
REM
REM 内容:
REM   Keiba-FeaturesIntegrityCheck: 毎日 22:00 起動
REM   features_integrity_monitor.py を read-only で実行、
REM   red flag 検出時のみ Discord (DISCORD_WEBHOOK_UPDATES) に通知
REM
REM 確認 cmd:
REM   schtasks /Query /TN "Keiba-FeaturesIntegrityCheck"
REM
REM 解除 cmd:
REM   schtasks /Delete /TN "Keiba-FeaturesIntegrityCheck" /F
REM ============================================================

setlocal

set TASKNAME=Keiba-FeaturesIntegrityCheck
set PYEXE=C:\Users\takum\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT=C:\Users\takum\keiba-ai\tools\features_integrity_monitor.py
set LOG=C:\Users\takum\keiba-ai\logs\features_integrity_%%date:~0,4%%%%date:~5,2%%%%date:~8,2%%.log
set CMD=%PYEXE% %SCRIPT% ^>^> %LOG% 2^>^&1

echo Registering schtask "%TASKNAME%"...
echo   command: %CMD%
echo   schedule: DAILY @22:00
echo.

schtasks /Create /TN "%TASKNAME%" /TR "cmd /c %CMD%" /SC DAILY /ST 22:00 /F

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: schtasks /Create failed. ADMIN 権限で実行する必要があります。
    exit /b 1
)

echo.
echo Registered successfully.
echo Verifying...
schtasks /Query /TN "%TASKNAME%"

endlocal
