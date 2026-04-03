@echo off
REM Register race auto-notify tasks (Sat/Sun 09:30)
REM Run as Administrator

echo Setting up Race Auto-Notify tasks...

schtasks /create /tn "keiba-ai\RaceAutoNotify_Sat" /tr "C:\Users\takum\keiba-ai\race_auto_notify.bat" /sc weekly /d SAT /st 09:30 /rl HIGHEST /f
schtasks /create /tn "keiba-ai\RaceAutoNotify_Sun" /tr "C:\Users\takum\keiba-ai\race_auto_notify.bat" /sc weekly /d SUN /st 09:30 /rl HIGHEST /f

if %errorlevel% equ 0 (
    echo Tasks created:
    echo   keiba-ai\RaceAutoNotify_Sat - Saturday 09:30
    echo   keiba-ai\RaceAutoNotify_Sun - Sunday 09:30
) else (
    echo Failed. Run as Administrator.
)
pause
