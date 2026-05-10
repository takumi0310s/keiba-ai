# Phase 14 D: paper trade setup

**作成**: 2026-05-10 (Session #90 Phase 14 D)
**目的**: V15 production と並行で V18/V20 を実弾なし shadow 評価する基盤を整備、 5/17 paper trade GO

---

## 1. paper_trade_engine.py 設計

### 1.1 file

`tools/paper_trade_engine.py` (新規、 約 230 行、 Phase 14 で作成)

### 1.2 責務

- V15 daily_predictions/ を mirror して paper_trade/v15_*.csv を出力 (production の hit/payout を信頼)
- V18 / V20 shadow predictions が paper_trade/v18_YYYYMMDD.csv 等にあれば evaluate
- daily_results/ と突合し model 別 hit / payout / pnl / ROI を計算
- summary_YYYYMMDD.csv (per-date) + summary_rolling.csv (cumulative) を出力

### 1.3 read-only 保証

- V15 production (predict_core / daily_predict / app.py) を一切変更しない
- 既存 daily_predictions / daily_results を read のみ
- 出力は data/paper_trade/ 配下のみ

### 1.4 使い方

```bash
# 単日 (5/10、 V15 のみ)
python tools/paper_trade_engine.py --date 20260510 --models v15

# 単日 (3 model、 V18/V20 shadow があれば自動 evaluate)
python tools/paper_trade_engine.py --date 20260510 --models v15,v18,v20

# 累計
python tools/paper_trade_engine.py --rolling
```

### 1.5 動作確認 (5/10)

```
=== paper_trade summary for 20260510 ===
model  n_races  n_hits  hit_rate  investment  payout    pnl    roi_pct
  v15       34      11   32.35%     23,800     27,090  +3,290  113.82%
  v18        0       0       NaN          0          0       0     NaN  (shadow 未生成)
  v20        0       0       NaN          0          0       0     NaN  (shadow 未生成)
```

→ V15 ROI 113.8% (5/10 単日)、 V18/V20 shadow predictions 生成は 別 session で対応

---

## 2. V18 shadow prediction 生成 plan (5/11-5/16 で別 session)

### 2.1 必要な作業

1. predict_core.py の build_features() 結果を v18_features 拡張版へ変換 (sib_w5 merge)
2. V18 LGB model (data/v18/v18v19_sib_exp_w5/v18_lgb_sib_exp_w5.txt) で predict
3. trio_bets を V15 と同じ 7 点で生成
4. data/paper_trade/v18_YYYYMMDD.csv に出力 (V15 daily_predictions と同 schema)

### 2.2 工数 (別 session)

| step | 推定 | 備考 |
|------|------|------|
| feature pipeline 整備 (sib_w5 merge) | 30-60 分 | jrdb_features.py + sib expanding csv 結合 |
| V18 inference script (tools/v18_predict.py) | 30-45 分 | predict_core 流用 |
| 5/10 retro 実行 + 検証 | 30 分 | V15 と diff 取得 |
| **合計** | **1.5-2.5 時間** | 別 session で 5/16 までに完了 |

---

## 3. V20 shadow prediction 生成 plan (5/24+ Phase 3 後半)

### 3.1 前提

- V20 PoC v2 (TFJV + JRDB + netkeiba マスター 統合) 学習完了
- 4-model ensemble (LGB+XGB+FT+IR) 構築完了
- WF 6-fold AUC ≥ 0.88 確認

### 3.2 工数 (Phase 3 後半 で集中投入)

| step | 推定 |
|------|------|
| V20 features 整備 (TFJV + JRDB + netkeiba マスター) | 5-8 時間 |
| 4-model ensemble 学習 (CPU) | 4-8 時間 |
| WF 6-fold 検証 | 2-3 時間 |
| LIVE retro + paper trade pipeline | 2-3 時間 |
| **合計** | **約 13-22 時間 (Phase 3 後半 1 週間で完了)** |

---

## 4. 5/17 (土) paper trade GO 条件

| 条件 | 状態 | 備考 |
|------|------|------|
| ✅ paper_trade_engine.py | ✅ 完了 (Phase 14) | tools/paper_trade_engine.py |
| ✅ V18 model file | ✅ 完了 (Session #43 C、 5/8) | data/v18/v18v19_sib_exp_w5/ |
| ⚠ V18 5/10+ shadow predictions | ⚠ 未生成 | 5/11-5/16 別 session で対応 |
| ⚠ V20 model | ⚠ PoC のみ | 5/24+ Phase 3 後半 で 4-model |
| ✅ daily_results 5/10 | ✅ 既存 | data/daily_results/20260510.csv |
| ✅ rolling summary CSV | ✅ 自動生成 | data/paper_trade/summary_rolling.csv |

→ **5/17 (土) は V18 paper trade 開始可能** (V20 は 5/24+)

---

## 5. 累計 paper trade 集計 plan

### 5.1 daily 自動集計 (Windows タスクスケジューラ 候補)

```bat
:: data/paper_trade_engine_daily.bat (新規候補)
@echo off
cd /d %~dp0..
python tools/paper_trade_engine.py --date %DATE_TODAY% --models v15,v18,v20
python tools/paper_trade_engine.py --rolling
```

→ daily_results.bat (20:00) 後に実行する schtasks 追加 (Phase 14 では設計のみ、 実装は別 session)

### 5.2 weekly_report.py との統合 (将来)

- weekly_report.py に "model 別 比較" section 追加
- V15 vs V18 vs V20 の週次 ROI 比較
- Discord #アップデート へ post

---

## 6. V15 投資保護 (絶対遵守)

✅ paper_trade_engine.py は read-only (production data 不変)
✅ V18/V20 shadow predictions は data/paper_trade/ 配下のみ
✅ V15 daily_predict / daily_results / app.py 一切変更なし
✅ 累計収支 +¥14,140 維持

---

## 7. 結論

✅ paper_trade_engine.py 整備完了 (本 Phase 14 B)
✅ V15 5/10 動作確認済 (ROI 113.8%)
⚠ V18 inference pipeline は 5/11-5/16 別 session で対応
⚠ V20 4-model ensemble は 5/24+ Phase 3 後半
✅ 5/17 (土) V18 paper trade 開始 ready

---

**Phase 14 D 完了** (Opus 4.7)
