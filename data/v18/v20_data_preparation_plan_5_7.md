# V20 学習 data 準備 plan (Session #41 E、 Phase 3 前倒し)

**作成**: 2026-05-08 深夜 (Session #41 E、 ユーザー就寝中)
**目的**: V20 (Phase 3 後半 6/9-30) 学習 data 主軸を JV-Link に切替 ための bulk fetch plan
**ファイル**: `tools/jvlink_full_backfill.py` (新規、 約 170 行)

---

## 1. 取得対象

| 項目 | 値 |
|------|----|
| 期間 | 2020/01/01 - 2025/12/31 (6 年分) |
| datatypes | RACE / SE / HR / O1 / TCOV / WOOD / BLOD = 7 種 |
| 推定 file 数 | 6 年 × 200 開催日 × 7 datatype = 約 8,400 fetches |
| 推定 records | RACE 100K, SE 1M, HR 30K, O1 500K, TCOV 1M, WOOD 200K, BLOD 80K = 約 3M+ |

---

## 2. 推定 size

| datatype | 推定 size (raw) | 推定 size (圧縮) |
|---------|---------------|----------------|
| RACE | 5-8 GB | 3-5 GB |
| SE | 15-25 GB | 10-15 GB |
| HR | 1-2 GB | 0.5-1 GB |
| O1 | 8-15 GB | 5-10 GB |
| TCOV | 5-10 GB | 3-7 GB |
| WOOD | 2-5 GB | 1-3 GB |
| BLOD | 0.5-1 GB | 0.3-0.6 GB |
| **計** | **36-66 GB** | **23-42 GB** |

→ 1 TB SSD 想定なら 余裕 (4-6.5%)、 容量 OK

---

## 3. fetch 工数見積

| 単位 | 件数 | 1 件 平均時間 | 累計時間 |
|------|------|-----------|--------|
| 月単位 | 72 か月 × 7 datatype = 504 fetch | 30s | 約 4h |
| 日単位 | 2,200 日 × 7 datatype = 15,400 fetch | 30s | 約 128h |

→ **月単位 fetch 推奨** (4h 以内に完了)

ただし JV-Link 仕様上、 大量 records は 1 fetch で 10 分以上かかる可能性:
- 月単位 SE (出走数多): 1 月分で 20-30 分の可能性
- 月単位 O1: 1 月分で 30-60 分の可能性
- → 実際は 8-12h 程度想定 (1 datatype × 月 × 平均 5-10 分)

→ 1 晩 (8h) では完了しない → **段階的 fetch 推奨**

---

## 4. tools/jvlink_full_backfill.py 機能

### 4.1 主要機能

- 月単位 / 日単位 fetch (`--monthly`)
- resume mode (`--resume`、 既 fetch を skip)
- max_runs 制限 (1 回の実行で fetch する run 数上限)
- datatypes 指定 (`--datatypes RACE,SE,HR`)
- dry-run (`--dry-run`)

### 4.2 使い方

```powershell
# 1. plan 確認
python tools\jvlink_full_backfill.py --from 20250101 --to 20251231 --monthly --dry-run

# 2. 1 か月分 fetch (試行)
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20250101 --to 20250131 --datatypes RACE

# 3. resume mode で続き fetch (途中失敗時)
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20200101 --to 20251231 --resume --max-runs 200 --monthly

# 4. 夜間 自動化 (schtasks)
schtasks /Create /TN "Keiba-JvlinkBackfillNightly" `
    /TR "powershell -File C:\Users\takum\keiba-ai\jvlink_nightly.ps1" `
    /SC DAILY /ST 23:00 /F
```

### 4.3 自動化 script (jvlink_nightly.ps1) 案

```powershell
# Phase 3 前半 (5/24-6/8) 期間中、 毎晩 23:00 に 200 件 fetch
cd C:\Users\takum\keiba-ai
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20200101 --to 20251231 --datatypes RACE,SE,HR,O1 `
    --resume --max-runs 200 --monthly
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\notify_done.py `
    "JVLink Nightly Backfill" "200 fetch 完了"
```

→ 6/9-13 まで に 6 年分 backfill 完了見込

---

## 5. data quality 検証 plan

### 5.1 既存 jra_races_full との照合

```python
# 公式 JV-Link RACE → DataFrame
official = parse_jvlink_race(date='20251231')
# 既存 jra_races_full.csv (netkeiba scraping 由来)
existing = pd.read_csv('data/jra_races_full.csv', dtype={'race_id':str})
# race_id / 距離 / 馬場 / 開催 で合致確認
diff = official.merge(existing, on='race_id', suffixes=('_jv','_nk'))
mismatches = diff[diff['distance_jv'] != diff['distance_nk']]
```

→ 不整合 件数 の妥当性 (期待 < 0.1%)

### 5.2 jrdb との照合

JRDB BAC ファイルとの比較:
- race 開催情報 (course, kai, nichi, race_num) 一致確認

### 5.3 V20 学習 data の最終整合性

V20 構築 (6/9-30) 直前に:
- JV-Link RACE+SE → V20 base data
- 既存 netkeiba speed_index / training_eval を補助 merge
- 不整合発見時、 公式 (JV-Link) 優先

---

## 6. data/jvlink_full/ 出力構造

```
data/jvlink_full/
├── RACE/
│   ├── 20200101_raw.csv
│   ├── 20200101_meta.json
│   ├── ...
├── SE/
├── HR/
├── O1/
├── TCOV/
├── WOOD/
└── BLOD/

合計 約 8,400 file 想定 (6 年 × 200 日 × 7 datatype)
```

→ `.gitignore` で除外推奨 (size 36-66 GB)

---

## 7. schedule (Phase 3 連携)

| 期間 | 内容 |
|------|------|
| 5/24 (土) | JRA-VAN 加入 + 32-bit Python install (manual) |
| 5/25-26 | 5/1-5/7 backfill (Session #41 C で plan 済) |
| 5/27-6/8 | 月単位 backfill (1 か月ずつ、 schtasks Nightly) |
| 6/9-6/13 | 全 6 年分 完了確認 + parser 完成 |
| 6/14-6/20 | V20 学習 data spec 確定 + 学習 |
| 6/21-6/25 | V20 WF 検証 |
| 6/26-6/30 | V20 GO/no-go + production 統合 |

---

## 8. risk + mitigation

| risk | mitigation |
|------|----------|
| 1 fetch で 30 分以上 | timeout + 月単位 fetch 推奨 |
| sleep 中の fetch 中断 | resume mode で再開 |
| size 想定超 (>100GB) | 段階的 削減 (priority datatype のみ fetch) |
| API rate limit | datatype 間 1s sleep + days 間 2s sleep |
| data 不整合 | quality check + 公式 (JV-Link) 優先 ルール |

---

## 9. 5/9 V15 投資保護 (E 領域)

✅ 既存 V15 production 完全不変
✅ data/jvlink_full/ は新規 path、 既存 data に影響なし
✅ 32-bit venv (別環境) で実行
✅ Session #41 E は plan + script のみ、 実行は 5/24+ ユーザー manual

→ **5/9 朝 V15 完全保証**

---

## 10. 結論

✅ E1: full backfill plan + size 試算 (36-66 GB raw)
✅ E2: tools/jvlink_full_backfill.py (170 行、 月単位/日単位、 resume、 max_runs)
✅ E3: data quality check plan (jra_races_full + jrdb 照合)
✅ E4: 統合 doc (本ファイル)

→ **Phase 3 前半 (5/24-6/8) 期間で 6 年分 backfill 段階的着手可**

---

**Session #41 E 完了**
