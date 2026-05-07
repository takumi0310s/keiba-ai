# JV-Link 5/1-5/7 backfill plan + script (Session #41 C)

**作成**: 2026-05-08 深夜 (Session #41 C、 ユーザー就寝中)
**前提**: A (32-bit Python) + B (jvlink_fetcher_v2.py) 完了
**ファイル**: `tools/jvlink_backfill_5_1_5_7.py` (新規、 約 130 行)

---

## 1. 取得対象

期間: 5/1 (金) - 5/7 (水) = 7 日
datatype: RACE / SE / HR / O1 = 4 種
合計 fetch: 7 × 4 = 28 runs

| date | RACE 想定 file 数 | SE 想定 | HR 想定 | O1 想定 |
|------|-----------------|--------|--------|--------|
| 5/1 (金) | 0-5 (平日、 重賞無し) | 0-5 | 0-5 | 0-5 |
| 5/2 (土) | 30-40 | 同 | 同 | 同 |
| 5/3 (日) | 30-40 (Session #40 で 29 件確認) | 同 | 同 | 同 |
| 5/4 (月祝) | 30-40 (祝日開催) | 同 | 同 | 同 |
| 5/5 (火祝) | 30-40 (祝日) | 同 | 同 | 同 |
| 5/6 (水) | 0-5 | 同 | 同 | 同 |
| 5/7 (木) | 0-5 | 同 | 同 | 同 |
| **計** | **約 130-180** | 同 | 同 | 同 |

合計推定 records: 4 datatype × 約 700-1000 records/day × 4 開催日 = 約 12,000-16,000

---

## 2. 実行 plan

### 2.1 dry-run (本 Session で確認済)

```bash
$ python tools/jvlink_backfill_5_1_5_7.py --dry-run
JV-Link backfill plan (20260501 - 20260507)
  dates (7): ['20260501', ..., '20260507']
  datatypes (4): ['RACE', 'SE', 'HR', 'O1']
  estimated runs: 28
  estimated time: 840s ~ 14min
[DRY-RUN] 実行 skip
  fetch RACE 20260501  ← 28 runs 表示
  ...
```

### 2.2 実行 (32-bit Python venv で)

```powershell
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_backfill_5_1_5_7.py `
    --from 20260501 --to 20260507 `
    --datatypes RACE,SE,HR,O1 `
    --parse
```

### 2.3 progress + summary 出力

各 day × datatype ごとに記録 + 最終 summary を `data/v18/jvlink_backfill_summary_20260501_20260507.json` に保存。

---

## 3. data quality check

### 3.1 record 数 妥当性

| datatype | 期待 records / day (G1 開催 weekend) | 期待 records / day (平日) |
|---------|-------------------------------------|------------------------|
| RACE | 60-80 (12 R × 6-7 column) | 0-30 |
| SE | 200-300 (12 R × 16-18 馬) | 0-150 |
| HR | 12-15 (12 R × 1 record) | 0-15 |
| O1 | 100-200 (時刻別 odds) | 0-50 |

→ 大幅な乖離 (期待値の 50% 未満 or 200% 超) なら 取得失敗 or 記録漏れ疑い

### 3.2 既存 jrdb data との整合性

5/3 (日) を例に:
- JV-Link RACE: 開催情報 (race_id, distance, surface, class)
- 既存 JRDB BAC260503: 同上
- → race_id / 距離 / 馬場 / 開催 が一致するか chunk-by-chunk diff

→ 不整合があれば JV-Link (公式) 優先。 既存 jrdb の bug 検出に。

### 3.3 5/2-5/3 USER 実投資との整合性

USER 申告:
- 5/2: 15R 1hit -9,350円 (投資 10,500円 = 700×15, 配当 1,150円)
- 5/3: 22R 4hits -7,950円 (投資 15,400円 = 700×22, 配当 7,450円)

JV-Link HR record で 各 race の trio 払戻を確認:
- USER 5/2 申告 hit が trio 配当と一致するか
- USER 5/3 申告 hit が trio 配当と一致するか

→ 不一致なら USER 集計 mistake or JV-Link data 不足の検出。

---

## 4. data 保存 path

```
data/jvlink/
├── RACE/
│   ├── 20260501_raw.csv
│   ├── 20260501_parsed.csv  (--parse 時)
│   ├── 20260501_meta.json
│   ├── 20260502_*  (同様)
│   ├── ...
├── SE/
│   ├── ...
├── HR/
│   ├── ...
└── O1/
    ├── ...
```

合計 file 数: 7 days × 4 datatypes × 3 file-types (raw/parsed/meta) = 84 files

---

## 5. 想定 size

| 出力 | size 想定 |
|------|---------|
| RACE/*.csv (raw + parsed) | 約 100-300 KB / day |
| SE/*.csv | 約 500 KB - 1 MB / day |
| HR/*.csv | 約 50-100 KB / day |
| O1/*.csv | 約 200-500 KB / day |
| **計 7 days × 4 dt** | **約 5-15 MB** |

→ git に commit 不要 (gitignore 推奨)、 ローカル data として保持。

---

## 6. 5/9 V15 投資保護 (C 領域)

✅ 既存 V15 production 完全不変
✅ data/jvlink/ は新規 path、 既存 data に影響なし
✅ 32-bit venv (別環境) で実行、 64-bit keiba-ai 環境 影響なし
✅ Session #41 C は plan + script のみ、 実行はユーザー manual

→ **5/9 朝 V15 完全保証**

---

## 7. 結論

✅ C1: backfill script `tools/jvlink_backfill_5_1_5_7.py` (130 行)
✅ C2: data quality check plan (record 数 + JRDB 整合 + USER 申告 整合)
✅ C3: 28 runs × 30s = 約 14 min 想定
✅ C4: 5-15 MB のローカル data 追加見込
✅ C5: 統合 doc (本ファイル)

→ **5/8 朝 ユーザー manual 実行で 5月分 backfill 即着手可**

---

**Session #41 C 完了**
