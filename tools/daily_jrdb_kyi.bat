@echo off
chcp 65001 >nul
setlocal

cd /d c:\Users\takum\keiba-ai

REM Windows 11 24H2+ deprecates wmic; use PowerShell
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set TODAY=%%I

set LOGFILE=logs\jrdb_kyi_auto_%TODAY%.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo [%date% %time%] Daily JRDB KYI Fetch Start (TODAY=%TODAY%) >> %LOGFILE%

%PYTHON_EXE% tools\scrape_jrdb.py --type KYI --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type SED --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type TYB --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type CYB --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type JOA --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\scrape_jrdb.py --type KAB --force --date %TODAY% >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\download_parse_jrdb_batch2.py --types kta cha >> %LOGFILE% 2>&1
%PYTHON_EXE% tools\download_parse_jrdb_extra.py --types kka jo >> %LOGFILE% 2>&1

REM OZ (基準オッズ: 単複・馬連) - 5/23 追加
%PYTHON_EXE% tools\download_parse_jrdb_batch2.py --types oz >> %LOGFILE% 2>&1
REM SR (ハロンタイム) - 5/23 追加
%PYTHON_EXE% tools\parse_jrdb_extended.py --types srb >> %LOGFILE% 2>&1
REM PACI は週次 (~2min) のため friday_weekend_scrape.bat に分離 (5/23)
REM python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1

echo [%date% %time%] Daily JRDB KYI Fetch Done >> %LOGFILE%
