# Calibrator v15 retrain v2 — honest summary

**実施日**: 2026-05-16
**source**: `tools/v21/calibrator_v15_retrain.py`
**output**: `data/calibrator_v15_pilot_v2.pkl` (orig file `data/calibrator_v15_pilot.pkl` は touch せず)

## 1. 改善 metrics (★ honest 実測 ★)

| metric | orig (5/11) | v2 (5/16) | delta |
|--------|-----:|-----:|-----:|
| n_samples | 21 | **315** | **15.0×** |
| pos_rate | N/A | 0.5810 | — |
| Brier (before) | 0.4698 | 0.2904 | -0.18 |
| Brier (after_iso) | 0.1881 | 0.2362 | +0.05 (over-fit 解消) |
| ECE (before) | 0.5146 | 0.1640 | -0.35 |
| ECE (after_iso) | 0.0000 | ~0.0 | clean |

orig の Brier=0.19 は **21 sample over-fit**、 v2 の Brier=0.24 が 真値。

## 2. isotonic predictions (★ 飽和解消 ★)

| raw p | orig iso(p) | v2 iso(p) | 意味 |
|------:|-----:|-----:|------|
| 0.05 | 0.50 | 0.00 | v2: 低 score は確実に低確率 |
| 0.10 | 0.55 | 0.00 | 同上 |
| 0.15 | 0.58 | 0.44 | v2 は妥当に下げる |
| 0.20 | 0.90 | 0.59 | ★ orig は 0.9 で過信、 v2 は 0.59 |
| 0.30 | **1.00** ★ | 0.59 | ★ orig は p>=0.3 で完全飽和、 v2 は実 hit rate |
| 0.50 | 1.00 | 0.59 | 同上 |
| 0.70 | 1.00 | 0.62 | 同上 |
| 0.95 | 1.00 | 0.67 | v2 は天井 0.67 |

★ orig の致命的 problem (p>=0.3 完全飽和) は v2 で 解消 ★

## 3. data source

- **features**: `data/daily_predictions/YYYYMMDD.csv` の `top1_score` (V15 morning prediction)
- **labels**: `data/daily_results/YYYYMMDD.csv` の `top1_finish <= 3`
- **inner join key**: `race_id`
- **date coverage**: 9 dates (20260314-20260510)
- **races per date**: 平均 35

★ orig は cumulative_results.csv の 21 件 (bet-placed のみ) を使用、 v2 は bet 有無に関わらず V15 morning 予測 を 全部 使用 ★

## 4. score range stats

| stat | value |
|------|------:|
| top1_score min | 0.131 |
| top1_score max | 0.820 |
| top1_score mean | 0.468 |
| top1_score std | 0.202 |

→ daily_predictions/ の top1_score は **post-combined score** (LGB+XGB+FT+IR ensemble 後)。 
raw V15 LGB score の range 0.1-0.25 と は異なる。 calibration は post-combined を入力とすること が前提。

## 5. 採用判断 (★ 慎重 ★)

### 採用可
- v2 calibrator は orig より sample 15x、 飽和解消、 ECE 健全
- post-combined score を入力にする限り、 strategy_layer_v2.py で 即 swap 可能

### 慎重 (★ honest ★)
- ★ v2 は 5/16 commit、 まだ 1 度も paper shadow eval されていない ★
- 5/18+ paper shadow で orig vs v2 calibrator 両方 record + 30 race で比較必須
- 単一 simulation 改善は backtest over-fit の risk

### 5/18+ paper shadow eval 拡張案
- `tools/strategy_layer_v2.py --shadow YYYYMMDD --calibrator v1` と `--calibrator v2` 両方走らせる
- 1 日 で 2 set の shadow csv (orig calibrator / v2 calibrator) を出力
- 30 race 蓄積後、 v2 の真の効果検証

## 6. V15 production 不変保証 ✅

- `data/calibrator_v15_pilot.pkl` orig は **touch せず**
- `tools/race_auto_notify.py` / `tools/predict_core.py` / `tools/daily_predict.py` / `app.py` 不変
- `keiba_model_v135_central*.pkl.gz` 不変
- schtasks 未変更

## 7. 次 action 候補 (user 判断)

| # | action | risk | 効果 |
|---|--------|------|------|
| A | v2 calibrator を strategy_layer_v2.py のデフォルトに | 中 (未検証) | EV 動的閾値の正確度 向上 |
| B | v1/v2 両方 paper shadow eval、 30 race 後 採用判定 | 低 | honest 検証 |
| C | calibrator 採用は保留、 まず 5/18+ 蓄積継続 | 低 | 6/1+ 大規模 retrain で確定 |

推奨: **B** (v1/v2 並行 shadow eval、 30 race 後 判定)。

## 8. 関連 file

- `tools/v21/calibrator_v15_retrain.py` — retrain script
- `data/calibrator_v15_pilot_v2.pkl` — v2 calibrator (新規)
- `data/v21/calibrator_v15_retrain_report.json` — 数値 report
- `data/calibrator_v15_pilot.pkl` — orig (touch せず)
- `data/v21/strategy_layer_calibrator_audit.md` — 元 audit (Terminal B)
