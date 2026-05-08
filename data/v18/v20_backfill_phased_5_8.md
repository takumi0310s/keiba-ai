# V20 backfill 部分前倒し plan + B 5/1-5/7 actual (Session #42 B + G)

**作成**: 2026-05-08 (Session #42 B + G、 ユーザー仕事中)
**前提**: Session #39 B (jvlink_fetcher v2)、 Session #41 C (5/1-5/7 plan)、 Session #41 E (6 年分 plan)
**目的**: B = 5/1-5/7 backfill 実行 plan、 G = V20 構築用 6 年分 phased schedule

---

## 1. B — 5/1-5/7 actual backfill (ユーザー manual)

### 1.1 実行手順

```powershell
# 32-bit Python venv (Session #42 A docs/SETUP_PYTHON32_QUICKSTART.md)
# Step 1-3 完了後

cd C:\Users\takum\keiba-ai

# 5/1-5/7 期間 fetch (約 14 分)
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_backfill_5_1_5_7.py `
    --from 20260501 --to 20260507 `
    --datatypes RACE,SE,HR,O1 `
    --parse

# 結果サマリー確認
$summary = Get-Content data\v18\jvlink_backfill_summary_20260501_20260507.json | ConvertFrom-Json
Write-Host "total records: $($summary.total_records)"
Write-Host "elapsed: $($summary.elapsed_sec)s"
```

### 1.2 期待 output

| 期間 | datatype | 期待 records |
|------|---------|------------|
| 5/1 (金) | RACE/SE/HR/O1 | 0-50 (平日) |
| 5/2 (土) | RACE/SE/HR/O1 | 200-400 |
| 5/3 (日) | RACE/SE/HR/O1 | 400-600 (29 files 確認済) |
| 5/4 (月祝) | RACE/SE/HR/O1 | 200-400 |
| 5/5 (火祝) | RACE/SE/HR/O1 | 200-400 |
| 5/6 (水) | RACE/SE/HR/O1 | 0-50 |
| 5/7 (木) | RACE/SE/HR/O1 | 0-50 |
| **計** | | **約 1,000-2,000 records** |

### 1.3 data quality check

```powershell
# 5/2-5/3 USER 申告 (Discord) との照合
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" -c @"
import pandas as pd
import os
# 5/2 USER 申告: 15R 1hit -9,350円 (投資 10,500=700×15、 配当 1,150)
# 5/3 USER 申告: 22R 4hits -7,950円 (投資 15,400=700×22、 配当 7,450)

# JV-Link HR record で 各 race の trio 払戻
hr_5_2 = pd.read_csv('data/jvlink/HR/20260502_raw.csv')
hr_5_3 = pd.read_csv('data/jvlink/HR/20260503_raw.csv')
print(f'HR records 5/2: {len(hr_5_2)}')
print(f'HR records 5/3: {len(hr_5_3)}')
# raw record から trio 払戻 抽出 → USER 集計と比較
"@
```

→ JV-Link 公式 払戻 vs USER 集計 の整合確認。

---

## 2. G — V20 構築用 6 年分 phased backfill

### 2.1 schedule (Session #41 E plan の詳細化)

| 期間 | 内容 | 推定 fetch 時間 |
|------|------|--------------|
| 5/24-5/26 | 32-bit env + 5/1-5/7 backfill 試行 (Session #42 A+B) | 30 min |
| 5/27-6/8 | schtasks Nightly で 200 件/晩 × 12 晩 | 2,400 fetch、 約 18-20h 累計 |
| 6/9-6/13 | 残り fetch + parser 完成 + integrity check | 約 5 日 |
| **計** | **6 年分 約 8,400 fetch** | **約 30-50 日 (Nightly)** |

### 2.2 schtasks 自動化

```cmd
schtasks /Create /TN "Keiba-JvlinkBackfillNightly" ^
    /TR "powershell -File C:\Users\takum\keiba-ai\jvlink_backfill_nightly.ps1" ^
    /SC DAILY /ST 23:00 /F
```

`jvlink_backfill_nightly.ps1` (新規、 Session #42 A docs に記載):

```powershell
cd C:\Users\takum\keiba-ai
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20200101 --to 20251231 `
    --datatypes RACE,SE,HR,O1 `
    --resume --max-runs 200 --monthly --parse
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\notify_done.py `
    "JVLink Nightly Backfill" "200 fetch 完了 (resume mode)"
```

### 2.3 進捗 監視

```bash
# meta JSON の数で進捗推定
ls data/jvlink_full/RACE/*_meta.json | wc -l
# 目標: 約 2,200 file (6 年分の RACE only)
```

`tools/jvlink_full_backfill.py` の `--resume` で 既 fetch を skip → 中断・再開 OK

### 2.4 V20 構築 schedule (Session #41 H roadmap v2 §2)

| 期間 | 内容 |
|------|------|
| 6/9-6/13 | JV-Link 全 6 年分 backfill 完了確認 + parser 完成 |
| 6/14-6/20 | V20 学習 data spec 確定 + V20 v1 学習 (4-model ensemble) |
| 6/21-6/25 | V20 WF 検証 (6-fold) |
| 6/26-6/28 | V20 LIVE retro + paper trading |
| 6/29-6/30 | V20 GO/no-go 最終判定 |
| 7/1+ | V20 production deploy (段階投入) |

---

## 3. data quality check (B + G 共通)

### 3.1 JV-Link RACE vs 既存 jra_races_full.csv

```python
# 1 か月分 sample で整合性確認
official = parse_jvlink_race(date_range='20250501-20250531')
existing = pd.read_csv('data/jra_races_full.csv', dtype={'race_id': str})
existing = existing[(existing['date'] >= '20250501') & (existing['date'] <= '20250531')]

merged = official.merge(existing, on='race_id', suffixes=('_jv', '_nk'))
# distance / surface / num_horses 一致確認
mismatches = merged[merged['distance_jv'] != merged['distance_nk']]
print(f'mismatch: {len(mismatches)} / {len(merged)}')
```

期待: mismatch < 0.1% (公式 vs netkeiba scrape の typo / 表記揺れのみ)

### 3.2 JV-Link HR vs jra_payouts.csv

`jra_payouts.csv` は 4/6 で停止 → 4/6 までの整合確認 + 4/7 以降は JV-Link 優先採用

### 3.3 不整合 検出時 ルール

```
公式 (JV-Link) を真値、 既存 (netkeiba/JRDB) を補助 source 扱い
```

---

## 4. V20 学習 data 構造 (G)

### 4.1 主軸 source

```
V20 学習 master:
├── JRA subset (6 年分)
│   ├── 主: data/jvlink_full/RACE/*.csv (公式)
│   ├── 主: data/jvlink_full/SE/*.csv (馬毎)
│   ├── 補助: data/jrdb_kyi.csv (PRE_RACE features)
│   └── 補助: data/netkeiba_speed_index.csv (V12 で採用済)
└── NAR subset
    └── data/nar_all_races.csv (既存 NAR scrape)
```

### 4.2 V20 features (Session #41 H roadmap v2 §2.2)

```python
V20_FEATURES = (
    V15_BASE_FEATURES                         # 150
    + KKA_FEATURES                            # 16
    - SKB_LEAK_FEATURES                       # -10 (Session #38)
    + SRB_FEATURES                            # 8
    + ['sib_top3_rate_exp_w5', 'sib_shinba_wr_exp_w5',  # Session #42 F window=5
       'sib_total_recent_races_w5',
       'sib_recent_offspring_count_w5']        # 4
    + JV_LINK_NEW_FEATURES                    # 5-15 (paci 自前算出 等)
)
# 計 ~170-180 features
```

### 4.3 期待 AUC

```
V15 BT WF AUC: 0.8939
V20 想定:       0.880-0.895 (SKB除外 -0.005 + JV-Link 公式 +0.001 + sib_exp w=5 +0.002)
```

---

## 5. 5/9 V15 投資保護 (B + G 領域)

✅ B/G いずれも 32-bit venv (別環境) で動作、 keiba-ai 64-bit 不変
✅ predict_core / daily_predict / V15 model 不変
✅ schtasks 既存 task 不変 (新規 Keiba-JvlinkBackfillNightly は 5/24+ admin で追加)
✅ data/jvlink_full/ は新規 path、 既存 data に影響なし

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ B1: 5/1-5/7 actual backfill 実行 plan (32-bit venv で 14 分、 manual)
✅ B2: data quality check (USER 申告 5/2-5/3 vs JV-Link HR 整合)
✅ G1: V20 構築 6 年分 phased schedule (5/24-6/8 = 30-40 日 Nightly)
✅ G2: schtasks Keiba-JvlinkBackfillNightly plan
✅ G3: V20 学習 data 構造 (Session #41 H roadmap v2 §2 詳細化)
✅ G4: V20 期待 AUC 0.880-0.895
✅ V15 production 完全不変

→ **5/24+ Phase 3 前半で 即着手可能**

---

**Session #42 B + G 完了**
