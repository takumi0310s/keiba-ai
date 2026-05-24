@echo off
REM race_auto_notify 手動再起動スクリプト
REM 使い方: netkeiba が 400 から回復したら実行

cd /d C:\Users\takum\keiba-ai

echo === netkeiba 到達確認 ===
powershell -Command "try { $r = Invoke-WebRequest -Uri 'https://race.netkeiba.com/' -TimeoutSec 10 -UserAgent 'Mozilla/5.0'; Write-Host 'OK:' $r.StatusCode } catch { Write-Host 'NG:' $_.Exception.Message }"

echo.
echo === RaceAutoNotify_Sun 再起動 ===
schtasks /run /TN "\keiba-ai\RaceAutoNotify_Sun"
if %ERRORLEVEL% == 0 (
    echo OK: 再起動完了
) else (
    echo ERROR: 再起動失敗
)
