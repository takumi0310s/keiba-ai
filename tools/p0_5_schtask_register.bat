@echo off
:: ============================================================
:: P0-5 schtask register (Keiba-LiveOrchestrator-15min)
:: ★ 5/18+ user 判断後 admin で実行 ★
:: ★ 5/17 中の admin 実行は絶対なし ★
:: ============================================================

setlocal

:: 1. admin 権限 check
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] admin 権限必要、 「管理者として実行」 してください
    pause
    exit /b 1
)

:: 2. 既存 schtask conflict check
schtasks /Query /TN "Keiba-LiveOrchestrator-15min" >nul 2>&1
if %errorLevel% equ 0 (
    echo [WARN] Keiba-LiveOrchestrator-15min 既登録、 unregister 後再登録?
    set /p choice="y/n: "
    if /i "%choice%"=="y" (
        schtasks /Delete /TN "Keiba-LiveOrchestrator-15min" /F
    ) else (
        echo [INFO] 中止しました
        exit /b 0
    )
)

:: 3. 登録 (WEEKLY SAT,SUN 08:30、 admin / HIGHEST RUNLEVEL)
schtasks /Create ^
  /TN "Keiba-LiveOrchestrator-15min" ^
  /TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat" ^
  /SC WEEKLY /D SAT,SUN /ST 08:30 ^
  /RL HIGHEST /F

if %errorLevel% equ 0 (
    echo [OK] Keiba-LiveOrchestrator-15min 登録成功
    schtasks /Query /TN "Keiba-LiveOrchestrator-15min" /V /FO LIST | findstr "TaskName Schedule Start"
) else (
    echo [ERROR] 登録失敗、 errorlevel=%errorLevel%
)

endlocal
