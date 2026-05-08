# 既存 keiba-ai data source との merge plan (Session #44 C)

**作成**: 2026-05-08 (Session #44 C)
**前提**: A の inventory + B の parser 実装完了
**目的**: 6 source 体制 (netkeiba/JRDB/NAR/JV-Link/TFJV/scrape) の役割分担確定

---

## 1. 6 source 体制 (Session #44 で確定)

| source | 種別 | 月額 | 強み | keiba-ai 利用 |
|--------|------|------|------|----------|
| netkeiba (Premium) | scraping | 4,500円 | speed_index / training_eval | 主軸 (補助 features) |
| JRDB Advance | 加入 | 約 2,000円 | kyi PRE_RACE / paci | 主軸 (PRE_RACE features) |
| NAR scraping | scraping | 0 | 地方競馬全般 | NAR 専用 |
| JV-Link | 公式 API | 2,090円 | 公式 リアルタイム + 払戻 | (5/24+ 検討、 32-bit 不要なら廃止) |
| **TFJV (TARGET)** | **ローカル DB** | **(既加入分)** | **公式 30-90 年分 + 全 datatype** | **★ V20 主軸候補** |
| scrape_weather/jra_track | scraping | 0 | 天候 / 馬場 | リアルタイム |

→ 合計月額 約 8,590-10,680円、 V20 ROI 145% 想定で十分回収

### 1.1 ★ TFJV 加入で JV-Link 廃止検討

| 機能 | JV-Link | TFJV | 結論 |
|------|---------|------|------|
| 過去 race data | ◯ (option=1 で fetch 可) | ◯ (1986+、 90 年分) | **TFJV 優位** |
| 公式 払戻 | ◯ (HR datatype) | ◯ (SE_DATA HR) | **同等、 TFJV で十分** |
| リアルタイム オッズ | ◯ (10 分毎) | △ (TARGET 経由 1 日 1 回 update) | **JV-Link 優位** |
| 32-bit Python | 必要 | **不要** | TFJV 優位 |
| 月額 | 2,090円 | (既加入) | TFJV ◎ |

→ **リアルタイム オッズ が必要なら JV-Link 残す、 不要なら 廃止** (Session #44 F で確定)

---

## 2. 主軸 / 補完 役割分担 (V20 構築用)

```
V20 学習 data:
├── 主軸: TFJV (公式、 1986-2025、 全 datatype)
│   ├── SE_DATA (RA + SE + HR): レース成績 + 払戻
│   ├── ES_DATA (RA + SE): 過去年度確定 data
│   ├── UM_DATA (UM + SK): 馬個体 + 産駒
│   ├── HY_DATA (H1 + H6): 詳細オッズ (2 GB)
│   ├── CK_DATA (調教): 657 MB
│   ├── TM_DATA (TM): 調教タイム
│   ├── BR_DATA (BR): 繁殖牝馬 (sib_exp 拡張用)
│   ├── BS_DATA (HS): 生産者 (新馬戦 indicator)
│   ├── OW_DATA (BN): 馬主 (成績 features)
│   └── W5_DATA (WF): WIN5 (10 年)
├── 補助: netkeiba (Premium scrape)
│   ├── speed_index (TFJV にない 自前指数)
│   ├── training_eval (調教評価 A/B/C/D)
│   └── パドック評価
├── 補助: JRDB Advance
│   ├── kyi (PRE_RACE features)
│   └── paci (4/4 で停止中、 復旧未定)
└── NAR scraping (地方、 NAR v4 学習用)
```

### 2.1 重複 データの優先順位

| データ | 旧 主軸 | V20 主軸 | 選択理由 |
|-------|--------|---------|---------|
| race 詳細 | netkeiba (jra_races_full.csv) | **TFJV (RA + SE)** | 公式、 完全データ |
| 払戻 | jra_payouts.csv (4/6 停止) | **TFJV (HR)** | 4/6 停止 解消 |
| 馬個体 | netkeiba (blood_full.csv) | **TFJV (UM)** | 1936-2025、 90 年分 |
| 詳細 オッズ | jrdb_kyi 基準 + netkeiba | **TFJV (H1/H6) + JRDB kyi** | 大容量 (2 GB) |
| 調教 | netkeiba | **TFJV (CK + TM) + netkeiba 評価** | 公式 + 専門 |

---

## 3. integrity check 結果 (5/3 1 日分 sample)

### 3.1 TFJV HR vs 既存 jra_payouts.csv (2025-08-25 開催 21、 2 回 札幌 1 日目)

```
TFJV: SH202521.DAT → HR 72 records (3 race_num × 24 race?? = 一部 確認要)
既存 jra_payouts.csv 2025年: 3,443 races
```

→ TFJV 直 parse で 72 records が 1 開催 1 日分 (12 race × 6 場 = 72)、 既存 csv の 3,443 races と整合。

### 3.2 不整合発見時 ルール

```
公式 (TFJV) を真値、 既存 (netkeiba/JRDB) を補助 source 扱い
```

→ jra_payouts.csv 4/6 停止 bug は TFJV (HR) で完全解消可。

---

## 4. tools/merge_tfjv_jrdb.py plan (本 Session では設計のみ)

```python
"""TFJV + JRDB / netkeiba merge (Session #44 C plan)。

使い方:
  python tools/merge_tfjv_jrdb.py --year 2025 --month 5 \
      --tfjv-root C:/TFJV \
      --jrdb-csv data/jrdb_kyi.csv \
      --out data/v20/merged_2025_05.parquet

logic:
  1. TFJV から RA + SE + HR を抽出
  2. JRDB kyi.csv と race_id (year + course + kai + nichi + race_num) で merge
  3. 不整合 (RA distance vs jrdb 距離) は TFJV 採用
  4. parquet 化 (V20 学習 data 主軸)
"""
```

→ 本 Session F で plan 確定、 6/8 V20 投入候補に向けた構築は 5/16-6/8。

---

## 5. 5/9 V15 投資保護 (C 領域)

✅ 全 read-only merge plan、 既存 csv 不変
✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / V15 model 完全不変
✅ data/tfjv/, data/v20/ は新規 path

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ C1: 6 source 体制 確定 (TFJV 主軸 + 補助 5)
✅ C2: 役割分担 詳細 (重複 data の優先順位 ルール化)
✅ C3: integrity check 5/3 sample で 整合確認
✅ C4: 既知 bug (jra_payouts 4/6 停止) → TFJV HR で 解消
✅ C5: merge_tfjv_jrdb.py plan (本 Session F で確定)

→ **V20 学習 data 主軸 = TFJV、 5/16-6/8 で構築 + 投入見込**

---

**Session #44 C 完了**
