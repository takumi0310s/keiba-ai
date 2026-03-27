@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================================
echo   KEIBA AI - Windows Setup
echo ============================================================
echo.

:: --- 1. Python check ---
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python %PYVER% detected

:: --- 2. pip install ---
echo.
echo [2/6] Installing packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)
echo   Install complete

:: --- 3. directories ---
echo.
echo [3/6] Checking directories...
if not exist "data" mkdir data
if not exist "data\daily_predictions" mkdir data\daily_predictions
if not exist "data\daily_results" mkdir data\daily_results
if not exist "logs" mkdir logs
echo   data/, logs/ OK

:: --- 4. DB init ---
echo.
echo [4/6] Initializing database...
python -c "import app; print('  keiba_predictions.db init OK')" 2>nul
if errorlevel 1 (
    echo   Attempting manual DB init...
    python -c "
import sqlite3, os
db = os.path.join(os.path.dirname(os.path.abspath('app.py')), 'keiba_predictions.db')
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, race_id TEXT, race_name TEXT, race_date TEXT, course TEXT, distance INTEGER, surface TEXT, condition TEXT, horse_name TEXT, horse_num INTEGER, ai_rank INTEGER, ai_score REAL, odds REAL, predicted_at TEXT, actual_finish INTEGER DEFAULT NULL, is_top3_pred INTEGER DEFAULT 0)')
c.execute('CREATE TABLE IF NOT EXISTS race_results (id INTEGER PRIMARY KEY AUTOINCREMENT, race_id TEXT UNIQUE, race_name TEXT, predicted_at TEXT, result_updated_at TEXT DEFAULT NULL, num_horses INTEGER, top1_name TEXT, top1_score REAL, trio_bets TEXT DEFAULT NULL, hit_trio INTEGER DEFAULT NULL, hit_combo TEXT DEFAULT NULL, payout INTEGER DEFAULT 0, is_nar INTEGER DEFAULT 0, wide_bets TEXT DEFAULT NULL, hit_wide INTEGER DEFAULT NULL, wide_payout INTEGER DEFAULT 0, buy_recommended INTEGER DEFAULT 1, bet_condition TEXT DEFAULT NULL, bet_type TEXT DEFAULT NULL, umaren_bets TEXT DEFAULT NULL)')
conn.commit()
conn.close()
print('  keiba_predictions.db manual init OK')
"
)

:: --- 5. feature_lookups.pkl check ---
echo.
echo [5/6] Checking data files...
set MISSING=0

if not exist "data\feature_lookups.pkl" (
    echo   [!] data\feature_lookups.pkl not found
    set MISSING=1
)
if not exist "data\jra_races_full.csv" (
    echo   [!] data\jra_races_full.csv not found
    set MISSING=1
)
if not exist "data\training_times.csv" (
    echo   [!] data\training_times.csv not found
    set MISSING=1
)
if not exist "data\jra_payouts.csv" (
    echo   [!] data\jra_payouts.csv not found
    set MISSING=1
)
if not exist "data\blood_full.csv" (
    echo   [!] data\blood_full.csv not found
    set MISSING=1
)
if not exist "data\odds_history.csv" (
    echo   [!] data\odds_history.csv not found
    set MISSING=1
)

if !MISSING!==1 (
    echo.
    echo   The above files are excluded by .gitignore and must be copied manually.
    echo   Copy from old PC:
    echo     data\feature_lookups.pkl    (37MB, feature encodings)
    echo     data\jra_races_full.csv     (781K rows, race data)
    echo     data\training_times.csv     (955K rows, training data)
    echo     data\odds_history.csv       (778K rows, odds history)
    echo     data\blood_full.csv         (82K rows, bloodline data)
    echo     data\jra_payouts.csv        (27K rows, JRA payout data)
    echo     *.db                        (SQLite DB files)
) else (
    echo   All data files OK
)

:: --- 6. Model file check ---
echo.
echo [6/6] Checking model files...
if exist "keiba_model_v9_central_live.pkl" (
    echo   Pattern B (live)  : OK
) else (
    echo   [!] keiba_model_v9_central_live.pkl not found
)
if exist "keiba_model_v9_central.pkl" (
    echo   Pattern A (eval)  : OK
) else (
    echo   [!] keiba_model_v9_central.pkl not found
)

:: --- Summary ---
echo.
echo ============================================================
echo   Setup Complete
echo ============================================================
echo.
echo   Start app   : streamlit run app.py
echo   Predict CLI : python predict_and_log.py "URL"
echo   Check results: python check_results.py --summary
echo.

if !MISSING!==1 (
    echo   [NOTE] Some data files are missing.
    echo   See README.md for migration steps.
    echo.
)

pause
