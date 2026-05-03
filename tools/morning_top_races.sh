#!/bin/bash
# 朝 06:30 起動用 1コマンド自動化 (汎用版)
# Usage: bash tools/morning_top_races.sh [YYYYMMDD]
#   無引数: 今日の日付を使用
#   引数指定: 指定日 (テスト/手動再実行用)
#
# 流れ:
#   1. JRDB 用データ認証curl 取得
#   2. 7zip で .lzh 解凍
#   3. JRDB CSV parse
#   4. v15 全レース予測 (daily_predict.py)
#   5. v17_morning で 11R/12R のみ予測 (三連複7点固定)
#   6. Discord 通知 (V15 vs V17 軸比較)

set -e
cd /c/Users/takum/keiba-ai
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 日付決定: 引数 > 今日 (YYYYMMDD)
if [ $# -ge 1 ] && [ -n "$1" ]; then
    DATE="$1"
else
    DATE=$(date +%Y%m%d)
fi
# JRDB用 YYMMDD
JRDB_DATE="${DATE:2:6}"

LOG="logs/morning_top_races_${DATE}.log"
mkdir -p logs data/v17

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "🌅 朝 ${DATE} (JRDB:${JRDB_DATE}) v17 11R/12R 自動パイプライン"
echo "開始: $(date)"
echo "Log: $LOG"
echo "============================================================"

# === Step 0: 認証情報読み込み ===
JRDB_ID=$(grep "^JRDB_ID=" .env | cut -d= -f2 | tr -d '"' | tr -d "'" | tr -d '\r')
JRDB_PASSWORD=$(grep "^JRDB_PASSWORD=" .env | cut -d= -f2 | tr -d '"' | tr -d "'" | tr -d '\r')
if [ -z "$JRDB_ID" ] || [ -z "$JRDB_PASSWORD" ]; then
    echo "❌ ERROR: JRDB_ID / JRDB_PASSWORD が .env になし"
    python tools/notify_done.py "Morning ERROR" "JRDB認証情報なし" --color red 2>/dev/null || true
    exit 1
fi
echo "JRDB auth: ID=${JRDB_ID:0:4}**** / PW=****"

# === Step 1: 認証curl で JRDB .lzh 取得 ===
echo ""
echo "=== STEP 1: JRDB ${JRDB_DATE} 取得 ==="
mkdir -p data/jrdb_raw_authfix

TYPES="Bac Cha Cyb Hjc Jo Kab Kka Kyi Paci Sed Skb Tyb Ukc"

for type in $TYPES; do
    type_lower=$(echo $type | tr '[:upper:]' '[:lower:]')
    type_upper=$(echo $type | tr '[:lower:]' '[:upper:]')
    mkdir -p "data/jrdb_raw_authfix/${type_lower}"

    url="http://www.jrdb.com/member/data/${type}/${type_upper}${JRDB_DATE}.lzh"
    out="data/jrdb_raw_authfix/${type_lower}/${type_upper}${JRDB_DATE}.lzh"

    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 200 ]; then
        echo "  ${type_upper}${JRDB_DATE}.lzh: skip (already)"
        continue
    fi

    code=$(curl -s -u "${JRDB_ID}:${JRDB_PASSWORD}" -o "$out" -w "%{http_code}" "$url" || echo "ERR")
    size=$(stat -c%s "$out" 2>/dev/null || echo 0)
    echo "  ${type_upper}${JRDB_DATE}.lzh: HTTP=${code} Size=${size}B"
done

# === Step 2: 7zip 解凍 ===
echo ""
echo "=== STEP 2: 7zip 解凍 ==="
SEVEN_ZIP="/c/Program Files/7-Zip/7z.exe"
if [ ! -f "$SEVEN_ZIP" ]; then
    SEVEN_ZIP="$(which 7z 2>/dev/null || echo /c/Program\ Files/7-Zip/7z.exe)"
fi

mkdir -p data/jrdb/extracted

for type in $TYPES; do
    type_lower=$(echo $type | tr '[:upper:]' '[:lower:]')
    type_upper=$(echo $type | tr '[:lower:]' '[:upper:]')

    src="data/jrdb_raw_authfix/${type_lower}/${type_upper}${JRDB_DATE}.lzh"
    dst="data/jrdb/extracted/${type}/"
    mkdir -p "$dst"

    if [ -f "$src" ]; then
        size=$(stat -c%s "$src" 2>/dev/null || echo 0)
        if [ "$size" -gt 200 ]; then
            "$SEVEN_ZIP" e "$src" -o"$dst" -y >/dev/null 2>&1 && \
                echo "  ${type_upper}${JRDB_DATE}: OK" || \
                echo "  ${type_upper}${JRDB_DATE}: 解凍失敗"
        else
            echo "  ${type_upper}${JRDB_DATE}: skip (size=${size})"
        fi
    fi
done

echo ""
echo "=== Step 2b: python scrape_jrdb.py 補完 ==="
for jt in KYI SED TYB CYB JOA KAB; do
    python tools/scrape_jrdb.py --type $jt --force --date $DATE 2>&1 | tail -2 || true
done

# === Step 3: parse JRDB → CSV ===
echo ""
echo "=== STEP 3: parse JRDB → CSV ==="
echo "  --- KKA / JO ---"
python tools/download_parse_jrdb_extra.py --skip-download --types kka jo 2>&1 | tail -8 || true
echo "  --- SRB / CHA / KTA / KAA ---"
python tools/download_parse_jrdb_batch2.py --skip-download --types srb cha kta kaa 2>&1 | tail -10 || true
echo "  --- KYI 系 ---"
python tools/parse_jrdb.py 2>&1 | tail -5 || true

# === Step 4: V15 全レース予測 ===
echo ""
echo "=== STEP 4: V15 全レース予測 (daily_predict.py) ==="
python tools/daily_predict.py --date $DATE 2>&1 | tail -15

PRED_CSV="data/daily_predictions/${DATE}.csv"
if [ ! -f "$PRED_CSV" ]; then
    echo "❌ ERROR: $PRED_CSV が生成されていない"
    python tools/notify_done.py "Morning ERROR" "daily_predict.py 失敗 ${DATE}" --color red 2>/dev/null || true
    exit 1
fi
N_RACES=$(grep -c "^202" "$PRED_CSV" 2>/dev/null || echo 0)
echo "  V15予測: ${N_RACES}レース完了"

# === Step 5: V17_morning 11R/12R 予測 + Discord ===
echo ""
echo "=== STEP 5: V17_morning 11R/12R 予測 ==="
# predict_v17_top_races_5_3.py は --date 引数を受け付ける
python tools/predict_v17_top_races_5_3.py --date $DATE --race-nums 11 12 2>&1 | tail -50

echo ""
echo "============================================================"
echo "✅ 完了: $(date)"
echo "📊 結果:"
echo "  V15: data/daily_predictions/${DATE}.csv (${N_RACES} races)"
echo "  V17: data/v17/predictions_${DATE:0:4}_${DATE:4:2}_${DATE:6:2}_top_races.csv"
echo "  ログ: $LOG"
echo "============================================================"
