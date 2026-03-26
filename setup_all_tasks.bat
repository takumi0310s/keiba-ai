@echo off
REM ============================================================
REM keiba-ai タスクスケジューラ一括登録
REM 管理者権限で実行すること（右クリック→管理者として実行）
REM ============================================================

echo ============================================================
echo   keiba-ai Task Scheduler Setup
echo ============================================================

REM 1. 毎日AM3:00 プレミアムデータ事前取得
schtasks /create /tn "keiba-ai\DailyPremiumScrape" /tr "C:\Users\sato\keiba-ai\daily_premium_scrape.bat" /sc daily /st 03:00 /rl HIGHEST /f
echo   1. DailyPremiumScrape (Daily 03:00) ... done

REM 2. 毎日AM8:00 全レース予測
schtasks /create /tn "keiba-ai\DailyPredict" /tr "C:\Users\sato\keiba-ai\daily_predict.bat" /sc daily /st 08:00 /rl HIGHEST /f
echo   2. DailyPredict (Daily 08:00) ... done

REM 3. 土曜AM9:30 レース5分前自動予測
schtasks /create /tn "keiba-ai\RaceAutoNotify_Sat" /tr "C:\Users\sato\keiba-ai\race_auto_notify.bat" /sc weekly /d SAT /st 09:30 /rl HIGHEST /f
echo   3. RaceAutoNotify_Sat (Sat 09:30) ... done

REM 4. 日曜AM9:30 レース5分前自動予測
schtasks /create /tn "keiba-ai\RaceAutoNotify_Sun" /tr "C:\Users\sato\keiba-ai\race_auto_notify.bat" /sc weekly /d SUN /st 09:30 /rl HIGHEST /f
echo   4. RaceAutoNotify_Sun (Sun 09:30) ... done

REM 5. 土曜18:00 結果照合
schtasks /create /tn "keiba-ai\DailyResults_Sat" /tr "C:\Users\sato\keiba-ai\daily_results.bat" /sc weekly /d SAT /st 18:00 /rl HIGHEST /f
echo   5. DailyResults_Sat (Sat 18:00) ... done

REM 6. 日曜18:00 結果照合
schtasks /create /tn "keiba-ai\DailyResults_Sun" /tr "C:\Users\sato\keiba-ai\daily_results.bat" /sc weekly /d SUN /st 18:00 /rl HIGHEST /f
echo   6. DailyResults_Sun (Sun 18:00) ... done

REM 7. 毎晩20:00 結果照合（平日含む）
schtasks /create /tn "keiba-ai\DailyResultsEvening" /tr "C:\Users\sato\keiba-ai\daily_results.bat" /sc daily /st 20:00 /rl HIGHEST /f
echo   7. DailyResultsEvening (Daily 20:00) ... done

REM 8. 月曜AM8:00 週次レポート
schtasks /create /tn "keiba-ai\WeeklyReport" /tr "C:\Users\sato\keiba-ai\weekly_report.bat" /sc weekly /d MON /st 08:00 /rl HIGHEST /f
echo   8. WeeklyReport (Mon 08:00) ... done

echo.
echo ============================================================
echo   All tasks registered. Listing:
echo ============================================================
schtasks /query /fo TABLE /tn "keiba-ai\*"
echo.
pause
