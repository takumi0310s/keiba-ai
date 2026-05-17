@echo off
:: ============================================================
:: P0-5 schtask unregister (rollback)
:: ★ rollback 専用、 admin で実行 ★
:: ============================================================

setlocal

:: 1. admin 権限 check
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] admin 権限必要、 「管理者として実行」 してください
    pause
    exit /b 1
)

:: 2. 存在 check
schtasks /Query /TN "Keiba-LiveOrchestrator-15min" >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Keiba-LiveOrchestrator-15min は登録されていません
    exit /b 0
)

:: 3. 削除
schtasks /Delete /TN "Keiba-LiveOrchestrator-15min" /F

if %errorLevel% equ 0 (
    echo [OK] Keiba-LiveOrchestrator-15min 削除成功
) else (
    echo [ERROR] 削除失敗、 errorlevel=%errorLevel%
)

endlocal
