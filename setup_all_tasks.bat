@echo off
REM ============================================================
REM keiba-ai Task Scheduler - Register All Tasks
REM Run as Administrator (right-click -> Run as administrator)
REM ============================================================

echo ============================================================
echo   keiba-ai Task Scheduler Setup
echo ============================================================

REM 1. Daily 03:00 - Premium data pre-fetch
schtasks /create /tn "keiba-ai\DailyPremiumScrape" /tr "C:\Users\takum\keiba-ai\daily_premium_scrape.bat" /sc daily /st 03:00 /rl HIGHEST /f
echo   1. DailyPremiumScrape (Daily 03:00) ... done

REM 2. Daily 08:00 - Predict all races
schtasks /create /tn "keiba-ai\DailyPredict" /tr "C:\Users\takum\keiba-ai\daily_predict.bat" /sc daily /st 08:00 /rl HIGHEST /f
echo   2. DailyPredict (Daily 08:00) ... done

REM 3. Saturday 09:30 - Auto-predict and notify before each race
schtasks /create /tn "keiba-ai\RaceAutoNotify_Sat" /tr "C:\Users\takum\keiba-ai\race_auto_notify.bat" /sc weekly /d SAT /st 09:30 /rl HIGHEST /f
echo   3. RaceAutoNotify_Sat (Sat 09:30) ... done

REM 4. Sunday 09:30 - Auto-predict and notify before each race
schtasks /create /tn "keiba-ai\RaceAutoNotify_Sun" /tr "C:\Users\takum\keiba-ai\race_auto_notify.bat" /sc weekly /d SUN /st 09:30 /rl HIGHEST /f
echo   4. RaceAutoNotify_Sun (Sun 09:30) ... done

REM 5. Saturday 18:00 - Check results
schtasks /create /tn "keiba-ai\DailyResults_Sat" /tr "C:\Users\takum\keiba-ai\daily_results.bat" /sc weekly /d SAT /st 18:00 /rl HIGHEST /f
echo   5. DailyResults_Sat (Sat 18:00) ... done

REM 6. Sunday 18:00 - Check results
schtasks /create /tn "keiba-ai\DailyResults_Sun" /tr "C:\Users\takum\keiba-ai\daily_results.bat" /sc weekly /d SUN /st 18:00 /rl HIGHEST /f
echo   6. DailyResults_Sun (Sun 18:00) ... done

REM 7. Daily 20:00 - Check results (including weekdays)
schtasks /create /tn "keiba-ai\DailyResultsEvening" /tr "C:\Users\takum\keiba-ai\daily_results.bat" /sc daily /st 20:00 /rl HIGHEST /f
echo   7. DailyResultsEvening (Daily 20:00) ... done

REM 8. Monday 08:00 - Weekly report
schtasks /create /tn "keiba-ai\WeeklyReport" /tr "C:\Users\takum\keiba-ai\weekly_report.bat" /sc weekly /d MON /st 08:00 /rl HIGHEST /f
echo   8. WeeklyReport (Mon 08:00) ... done

echo.
echo ============================================================
echo   All tasks registered. Listing:
echo ============================================================
schtasks /query /fo TABLE /tn "keiba-ai\*"
echo.
pause
