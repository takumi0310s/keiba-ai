@echo off
chcp 65001 >nul
setlocal

cd /d c:\Users\takum\keiba-ai

REM Windows 11 24H2+ deprecates wmic; use PowerShell
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\jrdb_kyi_auto_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Daily JRDB KYI Fetch Start (TODAY=%TODAY%) >> %LOGFILE%

python tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type CYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type JOA --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\download_parse_jrdb_batch2.py --types kta cha >> %LOGFILE% 2>&1
python tools\download_parse_jrdb_extra.py --types kka jo >> %LOGFILE% 2>&1

echo [%date% %time%] Daily JRDB KYI Fetch Done >> %LOGFILE%
