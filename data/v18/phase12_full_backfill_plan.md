# Phase 12 PoC D: 5/24+ full backfill plan (5/10)

> Session #87 Phase 12 PoC D 領域
> ★ honest report ★ — Phase 12 PoC で確認した 残作業を 5/24+ に精緻化

---

## 1. 5/24+ 全体 schedule

| 期間 | 内容 | 工数 |
|------|------|------|
| 5/24 (土) 朝 | 32-bit Python venv setup (jvlink-venv) | 約 100 min |
| 5/24 (土) 昼 | TFJV parser v3 (RA full layout) | 約 100 min |
| 5/24 (土) 夜 | HY_DATA parse + integration | 約 120 min |
| 5/25 (日) 朝 | WE / WH parse | 約 120 min |
| 5/25 (日) 昼 | UM / SK / BR parse | 約 180 min |
| 5/25 (日) 夜 | 17 features 全 真値化 + 動作 test | 約 120 min |
| **合計** | **2 日 (週末) で完了** | **約 740 min (12-13h)** |

→ Phase 17b の 30 年 backtest と並行可能 (data parse vs model 学習で I/O 衝突なし)

---

## 2. JV-Link COM full backfill (32-bit Python venv 経由)

### 2.1 setup 詳細

```powershell
# 1. 32-bit Python 3.13 download
Invoke-WebRequest "https://www.python.org/ftp/python/3.13.0/python-3.13.0.exe" `
  -OutFile python-3.13-x86.exe

# 2. install (32-bit、 user only)
.\python-3.13-x86.exe /passive InstallAllUsers=0 PrependPath=0 `
  TargetDir="C:\Users\takum\python313-x86"

# 3. venv 作成
& "C:\Users\takum\python313-x86\python.exe" -m venv "C:\Users\takum\jvlink-venv"

# 4. 必要 package install
& "C:\Users\takum\jvlink-venv\Scripts\pip.exe" install pywin32

# 5. COM 登録
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" `
  "C:\Users\takum\jvlink-venv\Scripts\pywin32_postinstall.py" -install

# 6. JV-Link 動作確認
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_test_python32.py
```

### 2.2 30 年 fetch (5/25-5/31)

```powershell
# 過去 30 年 fetch (1996-2025)、 期間 分割
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" `
  tools\jvlink_full_backfill.py --from 19960101 --to 20251231 `
  --datatypes RACE,SE,HR,O1,O2,O5,WE,WH `
  --resume --rate-limit 1.0
```

期待:
- 30 年 = 約 30 万 R = 約 3-5 GB raw data
- fetch 速度: 1 R / 1 秒 = 約 30 万 秒 = ★ 約 80 時間 ★ (連続)
- 実際は 中断 + 再開 で 1 週間 程度
- 同時並行で TFJV parser でも data 取得可

---

## 3. 中断 + 再開 logic

### 3.1 既実装 (Session #41 B + jvlink_full_backfill.py)
- ✅ `--resume` flag (last_filetime meta JSON)
- ✅ rate limit (`--rate-limit 1.0` 秒/R)
- ✅ datatype 間 sleep
- ✅ progress log

### 3.2 robustness 追加 (5/24 朝 setup 時)
- network 切断時 retry (3 回、 30s sleep)
- 途中 progress を 1 時間毎に Discord 通知
- crash 時 自動 restart

---

## 4. 17 features 真値化 完成 plan

### 4.1 features 別 真値化 source (5/24+)

| # | feature | 真値 source | 工数 |
|---|---------|-----------|------|
| A1 | jv_tansho_odds_open | O1 (単複) | 30 min |
| A2 | jv_fukusho_low_open | O1 | 同上 |
| A3 | jv_umaren_top_odds | O2 | 30 min |
| A4 | jv_trio_top_odds | O5 | 60 min (parser 新設) |
| B1 | jv_race_class_detail | RA full layout (race_name) | RA parser v3 完了で済 |
| B2 | jv_prize_structure_total | RA (賞金 1-5 fields) | 同上 |
| B3 | jv_entry_condition_enc | BT (番組テーブル) | 60 min (parser 新設) |
| C1 | jv_lap_first3f_pred | SE (前走、 区間タイム) | 60 min |
| C2 | jv_lap_last3f_pred | SE | 同上 |
| C3 | jv_race_pace_index | C1 / C2 計算 | 30 min |
| D1 | jv_baba_moisture | WE (馬場含水率) | 60 min |
| D2 | jv_baba_difference | WE (馬場差) | 同上 |
| D3 | jv_weather_change_score | WH (天候変化) | 60 min |
| E1 | jv_sire_dist_apt_score | UM + SK 集計 | 90 min |
| E2 | jv_dam_sire_apt_score | UM + BR 集計 | 60 min |
| E3 | jv_sire_surface_apt_score | UM + SK | 30 min |
| E4 | jv_ped_score_blend | E1-3 計算 | 10 min |
| **合計** | — | — | **約 670 min** |

### 4.2 完成判定 (5/26 月曜朝)
- 17/17 features 真値化 動作確認
- self-test pass (skeleton + 真値 切替 logic OK)
- 既 backfill 288 R で 17/17 真値抽出成功

---

## 5. Phase 17b と並行運用

| Phase 17b 作業 | Phase 12 PoC 残作業 | 並行可? |
|---------------|---------------------|--------|
| 30 年 backtest WF (RTX 4070 Ti SUPER) | TFJV parser v3 (CPU、 ファイル I/O) | ✅ |
| V20 学習 (GPU 集約) | JV-Link COM fetch (CPU、 network) | ✅ |
| paper trade test | data parse + 真値化 | ✅ |

→ 5/24-5/26 は Phase 17b と Phase 12 PoC を 同時並行 実施可。

---

## 6. 撤退条件 (絶対遵守)

5/24-5/26 で:
- 32-bit venv setup 失敗 → JV-Link COM 経路 中止、 TFJV 直 parse 一本化
- TFJV parser v3 で race_name 抽出 失敗 → race_class_detail 真値化 諦め (skeleton 維持)
- 累計 -¥30,000 → 5/26+ 全作業 halt、 V15 単独運用継続

---

## 7. V15 投資保護

✅ predict_core.py / V15 model 不変
✅ 32-bit venv は別 path (C:\Users\takum\jvlink-venv\)
✅ TFJV parse 出力は data/tfjv/ + data/jvlink/ 限定
✅ 累計 +¥14,140 維持

---

## 8. 結論

✅ D1: 5/24+ schedule 確定 (2 日週末で完了)
✅ D2: 32-bit venv setup plan (100 min)
✅ D3: TFJV parser v3 plan (100 min) + HY/WE/WH/UM/SK/BR parse plan (合計 670 min)
✅ D4: 中断 + 再開 logic (既実装)
✅ D5: Phase 17b 並行可 (I/O 衝突なし)
✅ D6: 撤退条件 明示

→ ★ 5/24-5/26 (週末 2 日) で 17/17 features 真値化 完成 想定 ★
