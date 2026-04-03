@echo off
REM Register daily results check tasks (Sat/Sun 18:00)
REM Run as Administrator

echo Setting up Daily Results tasks...

schtasks /create /tn "keiba-ai\DailyResults_Sat" /tr "C:\Users\takum\keiba-ai\daily_results.bat" /sc weekly /d SAT /st 18:00 /rl HIGHEST /f
schtasks /create /tn "keiba-ai\DailyResults_Sun" /tr "C:\Users\takum\keiba-ai\daily_results.bat" /sc weekly /d SUN /st 18:00 /rl HIGHEST /f

if %errorlevel% equ 0 (
    echo Tasks created:
    echo   keiba-ai\DailyResults_Sat - Saturday 18:00
    echo   keiba-ai\DailyResults_Sun - Sunday 18:00
) else (
    echo Failed. Run as Administrator.
)
pause
