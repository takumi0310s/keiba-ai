# 5/23 Feature Fetch Fix — OZ / SR / PACI 取得再開

Date: 2026-05-23

## Summary

PACI (4/4 停止) / OZ (3/31 以降 CSV 未更新) / SR (未スケジュール) の取得を再開。
V15 予測・predict_core.py・daily_predict.py は完全不変。データ貯めるだけ。

---

## Script Test Results

### OZ (tools/download_parse_jrdb_batch2.py --types oz)

- Status: **OK**
- jrdb_oz.csv: 22,108 rows / 40 cols
- Latest race_id in CSV: **202610011212** (2026年10月01日 12R 12頭)
- Elapsed: ~27s
- 毎日実行可能 (冪等)

### SR (tools/parse_jrdb_extended.py --types srb)

- Status: **OK**
- jrdb_sr.csv: 39,208 rows / 32 cols
- Latest race_id: **202610011212**
- 2026 coverage: 1,206 rows (2015-2026 全年)
- Elapsed: ~3s
- 毎日実行可能 (冪等)

### PACI (tools/scrape_jrdb_paci.py)

- Status: **OK**
- jrdb_paci.csv: 551,148 rows / 63 cols (403KB 増加)
- 新規取得 ZIP: 4件 (PACI260516〜PACI260524)
- Elapsed: ~127s (重い → 週次 bat に分離)
- 2026 coverage: 19,529 rows
- dry-run 確認後、本番実行 → 成功

---

## V15 Prediction Impact Assessment

predict_core.py で `jrdb_oz` / `jrdb_sr` / `jrdb_paci` を CSV から直接 read_csv している箇所: **0件** (grep 確認済み)

PACI 特徴量 (`paci_sogo_mark` 等) は predict_core.py 内で `if _paci not in df.columns: df[_paci] = 0` でゼロ埋めされるだけ。
CSV が更新されても predict_core.py はこれらを読み直さないため、**V15 予測への自動流入なし**。

**結論: データは貯まるが V15 予測には自動反映されない (設計通り)。**

---

## Changes to Bat Files

### tools/daily_jrdb_kyi.bat (毎朝 自動実行)

追加 (既存 kka/jo fetch の直後):
```bat
REM OZ (基準オッズ: 単複・馬連) - 5/23 追加
python tools\download_parse_jrdb_batch2.py --types oz >> %LOGFILE% 2>&1
REM SR (ハロンタイム) - 5/23 追加
python tools\parse_jrdb_extended.py --types srb >> %LOGFILE% 2>&1
REM PACI は週次 (~2min) のため friday_weekend_scrape.bat に分離 (5/23)
REM python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1
```

OZ ~27s + SR ~3s = 毎日合計 +30s。許容範囲。

### friday_weekend_scrape.bat (毎週金曜 AM 10:00)

追加 (scrape_weekend_thisweek.py の直後):
```bat
REM PACI (前日データ) - 週次取得 (~2min) - 5/23 追加
python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1
```

PACI は週次レースデータ (前日 KYI 拡張) なので金曜取得が適切。

---

## Paper Verification Plan (5/24+)

現状:
- V15 予測は PACI/OZ/SR 欠損時はゼロ埋め (既存動作)
- 5/23 以降は各 CSV に最新データが入る

5/24+ で実施:
1. race_notify_log v2 で「V15 通常予測スコア」を記録
2. 別途、fresh PACI/OZ/SR を merge した場合の予測スコアを shadow で計算
3. 両スコアを比較 → 特徴量補完の効果を定量化
4. 効果確認後、V20 設計に反映

---

## Regression Test

```
V15 features: 145
V15 auc: 0.8939485520467574
V15 version: v15
V15 unchanged: PASS

predict_core import: PASS
```

V15 .pkl.gz / predict_core.py / daily_predict.py = 完全不変。
