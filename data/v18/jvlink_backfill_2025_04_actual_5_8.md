# JV-Link 2025-04 1 か月分 backfill plan (Session #43 B)

**作成**: 2026-05-08 (Session #43 B)
**前提**: Session #42 A docs/SETUP_PYTHON32_QUICKSTART.md で 32-bit env 構築済 (manual)
**目的**: V20 学習 data 主軸 切替の前段階として、 1 か月分 (2025-04) を実 fetch → 整合 check

---

## 1. 実行 plan

### 1.1 32-bit venv での実行 command

```powershell
cd C:\Users\takum\keiba-ai

# 2025-04 1 か月分 fetch (推定 60-90 min)
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20250401 --to 20250430 `
    --datatypes RACE,SE,HR,O1 `
    --parse `
    --monthly
```

### 1.2 期待 output

```
data/jvlink_full/
├── RACE/
│   ├── 20250401_raw.csv
│   ├── 20250401_parsed.csv
│   ├── 20250401_meta.json
│   ├── ... (30 日分)
├── SE/
├── HR/
└── O1/
```

合計 file 数: 30 days × 4 datatypes × 3 file types = 360 files
推定 size: 約 1-2 GB

---

## 2. 期待 records (4 月)

| date 種別 | n days | 期待 records (4 datatypes) |
|----------|--------|--------------------------|
| 平日 (約 21 日) | 21 | 0-50 records / day = 0-1,000 |
| 土曜 (5 日) | 5 | 約 600-1,000 / day = 3,000-5,000 |
| 日曜 (4 日) | 4 | 約 600-1,000 / day = 2,400-4,000 |
| 祝日 (0 日) | 0 | 0 |
| **計** | **30** | **約 5,400-10,000 records** |

---

## 3. data quality check (実行後)

### 3.1 既存 data との整合性確認

```python
import pandas as pd

# JV-Link RACE
jv_race = pd.read_csv('data/jvlink_full/RACE/20250419_raw.csv')
print(f'JV-Link 4/19 RACE records: {len(jv_race)}')

# 既存 jra_races_full.csv (netkeiba 由来)
nk = pd.read_csv('data/jra_races_full.csv', dtype={'race_id': str})
nk_4_19 = nk[(nk['year'] == 25) & (nk['month'] == 4) & (nk['day'] == 19)]
print(f'既存 4/19 race rows: {len(nk_4_19)}')

# race_id / 距離 / 馬場 一致確認 (parser 後)
# parser 完成後 (Phase 3 後半 6/9-13) に詳細 diff
```

### 3.2 既知 bug 解消確認

```python
# jra_payouts.csv 4/6 停止 → JV-Link HR で代替
hr = pd.read_csv('data/jvlink_full/HR/20250419_raw.csv')
print(f'JV-Link HR 4/19 records: {len(hr)}')
# parser で trio_payout 抽出 → 4/19 12 races 分の 払戻 確認
```

---

## 4. V20 期待 AUC 試算

### 4.1 1 か月分での V15 Pseudo-PoC

V20 学習 (Session #41 H roadmap v2) は 6 年分 (2020-2025) 想定、 **1 か月分では sample 不足** で正式学習不能。

ただし PoC として:
- 既存 V15 model に JV-Link RACE/SE 1 か月分を追加 features として merge
- 既存 V15 features (150) + JV-Link 5-15 = ~155-165 features
- 既存 V15 model を そのまま 適用 (再学習なし) で 4 月の予測精度確認

### 4.2 期待効果

```
V15 baseline (既存 4 月 OOS): AUC ≈ 0.89
V15 + JV-Link merge sample: AUC 0.89 + 0.001-0.005 程度 (1 か月では 微増)
```

→ 大規模効果は Phase 3 後半 (6 年分) で確認

---

## 5. Phase 3 schedule との整合 (Session #42 G、 #41 H)

| 期間 | 内容 |
|------|------|
| **5/24 (土) AM** | 32-bit Python install (admin、 約 15 分) |
| **5/24 (土) PM** | 5/1-5/7 backfill 試行 (約 14 分) |
| **5/25-5/26** | data quality check (5/2-5/3 USER 申告 vs HR 整合) |
| **5/27** | 2025-04 1 か月分 backfill (本 doc) |
| 5/28-6/8 | schtasks Nightly 200/晩 で 6 年分 段階 fetch |
| 6/9-6/13 | 全 6 年分 fetch 完了 + parser 完成 |
| 6/14-6/30 | V20 学習 + 検証 |
| 7/1+ | V20 production deploy |

---

## 6. 5/9 V15 投資保護 (B 領域)

✅ B は 5/24+ ユーザー manual 実行 plan のみ、 5/9 投資には影響なし
✅ V15 model file md5 不変
✅ predict_core / daily_predict / app.py / schtasks 既存 完全不変
✅ data/jvlink_full/ は新規 path、 既存 data に影響なし

→ **5/9 朝 V15 完全保証**

---

## 7. 結論

✅ B1: 2025-04 1 か月分 backfill 実行 plan (32-bit venv、 約 60-90 min)
✅ B2: 期待 records 5,400-10,000 (30 days × 4 datatypes)
✅ B3: data quality check (USER 申告 vs HR 整合 + 既存 jra_races_full との照合)
✅ B4: V20 期待 AUC 試算 (1 か月では 微増 +0.001-0.005、 大規模効果は 6 年分で)
✅ Phase 3 schedule との整合 確認

→ **5/24+ Phase 3 前半で 即着手可能**

---

**Session #43 B 完了**
