#!/bin/bash
set -e

echo "============================================================"
echo "  KEIBA AI - Mac/Linux Setup"
echo "============================================================"
echo ""

# --- 1. Python check ---
echo "[1/6] Python確認..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 が見つかりません。Python 3.10以上をインストールしてください。"
    exit 1
fi
PYVER=$(python3 --version 2>&1)
echo "  $PYVER 検出"

# --- 2. pip install ---
echo ""
echo "[2/6] パッケージインストール..."
pip3 install -r requirements.txt
echo "  インストール完了"

# --- 3. directories ---
echo ""
echo "[3/6] ディレクトリ確認..."
mkdir -p data/daily_predictions data/daily_results logs
echo "  data/, logs/ 確認OK"

# --- 4. DB init ---
echo ""
echo "[4/6] データベース初期化..."
python3 -c "
import sqlite3, os
db = os.path.join(os.path.dirname(os.path.abspath('app.py')), 'keiba_predictions.db')
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT, race_name TEXT, race_date TEXT,
    course TEXT, distance INTEGER, surface TEXT, condition TEXT,
    horse_name TEXT, horse_num INTEGER, ai_rank INTEGER,
    ai_score REAL, odds REAL, predicted_at TEXT,
    actual_finish INTEGER DEFAULT NULL, is_top3_pred INTEGER DEFAULT 0)''')
c.execute('''CREATE TABLE IF NOT EXISTS race_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT UNIQUE, race_name TEXT,
    predicted_at TEXT, result_updated_at TEXT DEFAULT NULL,
    num_horses INTEGER, top1_name TEXT, top1_score REAL,
    trio_bets TEXT DEFAULT NULL, hit_trio INTEGER DEFAULT NULL,
    hit_combo TEXT DEFAULT NULL, payout INTEGER DEFAULT 0,
    is_nar INTEGER DEFAULT 0, wide_bets TEXT DEFAULT NULL,
    hit_wide INTEGER DEFAULT NULL, wide_payout INTEGER DEFAULT 0,
    buy_recommended INTEGER DEFAULT 1, bet_condition TEXT DEFAULT NULL,
    bet_type TEXT DEFAULT NULL, umaren_bets TEXT DEFAULT NULL)''')
conn.commit()
conn.close()
print('  keiba_predictions.db 初期化OK')
"

# --- 5. Data file check ---
echo ""
echo "[5/6] 大容量データファイル確認..."
MISSING=0

check_file() {
    if [ ! -f "$1" ]; then
        echo "  [!] $1 が見つかりません"
        MISSING=1
    fi
}

check_file "data/feature_lookups.pkl"
check_file "data/jra_races_full.csv"
check_file "data/training_times.csv"
check_file "data/jra_payouts.csv"
check_file "data/blood_full.csv"
check_file "data/odds_history.csv"

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  上記ファイルは .gitignore 対象のため手動コピーが必要です。"
    echo "  旧PCから以下をコピーしてください:"
    echo "    data/feature_lookups.pkl    (37MB, 特徴量エンコーディング)"
    echo "    data/jra_races_full.csv     (781K行, レースデータ)"
    echo "    data/training_times.csv     (955K行, 調教データ)"
    echo "    data/odds_history.csv       (778K行, オッズ履歴)"
    echo "    data/blood_full.csv         (82K行, 血統データ)"
    echo "    data/jra_payouts.csv        (27K行, JRA配当データ)"
    echo "    *.db                        (SQLite DBファイル)"
else
    echo "  全データファイル確認OK"
fi

# --- 6. Model file check ---
echo ""
echo "[6/6] モデルファイル確認..."
[ -f "keiba_model_v9_central_live.pkl" ] && echo "  Pattern B (実運用) : OK" || echo "  [!] keiba_model_v9_central_live.pkl が見つかりません"
[ -f "keiba_model_v9_central.pkl" ] && echo "  Pattern A (評価用) : OK" || echo "  [!] keiba_model_v9_central.pkl が見つかりません"

# --- Summary ---
echo ""
echo "============================================================"
echo "  セットアップ完了"
echo "============================================================"
echo ""
echo "  アプリ起動: streamlit run app.py"
echo "  予測CLI  : python3 predict_and_log.py \"URL\""
echo "  結果照合  : python3 check_results.py --summary"
echo ""

if [ $MISSING -eq 1 ]; then
    echo "  [注意] 一部データファイルが不足しています。"
    echo "  README.md の「移行手順」を確認してください。"
    echo ""
fi
