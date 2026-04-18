@echo off
chcp 65001 >nul
setlocal

cd /d c:\Users\takum\keiba-ai

set LOGFILE=logs\jrdb_kyi_auto_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] Daily JRDB KYI Fetch Start >> %LOGFILE%

REM 今日分と明日分の両方を取得（翌日開催の前日情報はKYI{今日}.lzhに入る）
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set dt=%%I
set TODAY=%dt:~0,8%

python tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type CYB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type JOA --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1
python tools\download_parse_jrdb_batch2.py --types kta cha >> %LOGFILE% 2>&1
python tools\download_parse_jrdb_extra.py --types kka jo >> %LOGFILE% 2>&1

echo [%date% %time%] Daily JRDB KYI Fetch Done >> %LOGFILE%
