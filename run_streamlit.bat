@echo off
REM ==========================================================
REM Streamlit safe launcher — ポート8501の多重起動を防止
REM   既存プロセスがあれば警告して終了（--force で自動kill）
REM   使用:  run_streamlit.bat           ... 通常起動（警告のみ）
REM          run_streamlit.bat --force   ... 既存プロセスを自動kill
REM ==========================================================
setlocal

cd /d %~dp0

set PORT=8501
set FORCE=0
if "%1"=="--force" set FORCE=1

REM ---- ポート8501の使用状況チェック ----
set FOUND_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
    set FOUND_PID=%%a
)

if defined FOUND_PID (
    echo [WARN] ポート %PORT% は既にプロセス PID=%FOUND_PID% が使用中です。
    if %FORCE%==1 (
        echo [INFO] --force 指定のため既存プロセスを終了します...
        taskkill /F /PID %FOUND_PID%
        timeout /T 2 /NOBREAK >nul
    ) else (
        echo [ABORT] 既存のStreamlitを手動で停止するか、--force オプションで再実行してください。
        echo        例:  run_streamlit.bat --force
        exit /b 1
    )
)

echo [INFO] Streamlit を起動します ^(port=%PORT%^) ...
python -m streamlit run app.py --server.port %PORT%
endlocal
