@echo off
REM netkeiba 回復監視 + race_auto_notify 自動再起動
REM コマンドプロンプトで実行: tools\netkeiba_watchdog.bat

cd /d C:\Users\takum\keiba-ai

echo =====================================================
echo  netkeiba Watchdog 起動
echo  回復確認後 → RaceAutoNotify_Sun 再起動
echo  停止: Ctrl+C
echo =====================================================

:LOOP
for /f "tokens=1-2 delims=:" %%a in ("%time%") do (
    set HH=%%a
    set MM=%%b
)
set /a HH_NUM=%HH%
if %HH_NUM% GEQ 17 (
    echo [%time%] 17時以降 - レース終了。監視終了。
    exit /b 0
)

echo.
echo [%time%] netkeiba 確認中...
powershell -Command "$r=0; try { $resp=Invoke-WebRequest -Uri 'https://race.netkeiba.com/race/shutuba.html?race_id=202605021006' -TimeoutSec 10 -UserAgent 'Mozilla/5.0' -UseBasicParsing -ErrorAction Stop; $r=$resp.StatusCode } catch { $r=0 }; Write-Host $r" > %TEMP%\nk_status.txt 2>&1
set /p STATUS=<%TEMP%\nk_status.txt

echo [%time%] Status: %STATUS%

if "%STATUS%"=="200" (
    echo [%time%] ★ netkeiba 回復! race_auto_notify 再起動...
    schtasks /run /TN "\keiba-ai\RaceAutoNotify_Sun"
    if %ERRORLEVEL% == 0 (
        echo [%time%] 再起動成功! 監視終了。
        python tools/notify_done.py "watchdog" "netkeiba 回復 + RaceAutoNotify_Sun 再起動成功"
        exit /b 0
    ) else (
        echo [%time%] 再起動失敗。手動実行: schtasks /run /TN "\keiba-ai\RaceAutoNotify_Sun"
        exit /b 1
    )
)

echo [%time%] まだブロック中。3分後に再試行...
timeout /t 180 /nobreak > nul
goto LOOP
