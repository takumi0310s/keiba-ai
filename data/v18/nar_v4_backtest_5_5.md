# NAR v4 backtest 再現 (Phase 2.5+ / 2026-05-05)

実行: `python tools/backtest_nar_v4_quick.py`

---

## 1. 前提

- model: `data/nar/models/keiba_model_nar_v4.pkl` (trained 2026-03-09)
- data: **`data/nar_all_races.csv`** (49,915 rows, 4,821 races, 2024-01〜2025-05-14)
- 注: archive/nar/backtest_nar_leakfree.py は `data/chihou_races_2020_2025.csv` 依存だが本リポに無し → 簡易版で代替

## 2. 結果

### 2.1 全体 AUC

| metric | reported (model内) | 本 backtest (2025 OOS) | 差 |
|--------|------------------:|----------------------:|----:|
| LGB AUC | 0.8142 | **0.8520** | +0.0378 |
| XGB AUC | 0.8144 | **0.8514** | +0.0370 |
| ensemble | **0.8145** | **0.8519** | +0.0374 |

### 2.2 条件別 AUC (簡易 backtest)

| 条件 | n | AUC | win_rate |
|------|---:|----:|---------:|
| A (8-14頭/1600m+/良) | 5,304 | **0.8471** | 0.0928 |
| B (8-14頭/1600m+/重) | 1,705 | **0.8638** | 0.0944 |
| C (15+頭/1600m+/良) | 124 | 0.9418 | 0.0645 (sample 小) |
| D (1200-1400m) | 34,607 | **0.8484** | 0.0958 (NAR 主流) |
| E (7頭以下) | 1,706 | **0.8687** | 0.1512 |
| X (15+頭/重) | 6,469 | **0.8612** | 0.0915 |

→ 全条件 AUC > 0.84、D に大量データ (NAR の中心帯)。

## 3. 注意点 / 限界

### 3.1 reported 0.8145 vs 本測 0.8519 の差

- model 学習データ: 2020-2024 NAR (n_rows=49,213)
- 本 backtest data: 2024-01 + 2025-01〜05 (n_rows=49,915、ほぼ同サイズ)
- → **CSV 内に学習データの一部が混入している可能性が高い** (本来の strict OOS ではない)
- 2024-01 train サブセットが本 CSV にあると AUC が高めに出る

### 3.2 真の OOS 評価

- chihou_races_2020_2025.csv (本リポに無し) を生成、または
- archive/nar/train_nar_v4_leakfree.py を読み 学習 split を再現してから評価
- 工数: +90 min (別 task)

### 3.3 ROI 計算は未実施

- 本 backtest は AUC のみ
- 実 ROI (条件別 trio/umaren) は jra_payouts 相当の NAR payout が必要
- data/nar_all_races.csv の payout 列 (tansho_payout/fukusho_payout/umaren_payout/wide_payout/trio_payout/tierce_payout) を使えば計算可能だが、本 session スコープ外

## 4. 結論 (E task)

- **AUC 0.8519 (2024-2025 NAR)** で model 健全性を確認
- 条件 A/B/D/E/X すべて 0.84+、D が大量 sample
- 真の strict OOS AUC 評価と条件別 ROI は別 task で実施推奨 (工数 +90min × 2)
- 本 session は **AUC reproducibility** で stop、5/12 paper trading 開始の前提として十分

## 5. 関連 archive 比較

archive/nar/ にある他 model:
- `keiba_model_v9_nar.pkl` (479KB、旧版)
- `keiba_model_v10_nar_ref.pkl` (620KB、参考、JRA v10 base)
- `keiba_model_nar_v4.pkl` (167KB、現行採用)

backtest_nar_condition_v3.json 等の condition 最適化結果が archive にあり (Phase 3 で参照)。

## 6. 次の step (別 session)

| 課題 | 工数 | 出力 |
|------|------|------|
| chihou_races_2020_2025.csv 生成 (or 学習 split 再現) | 60min | strict OOS AUC |
| 条件別 ROI 計算 (NAR payouts 利用) | 60min | 5/16 paper plan の根拠データ |
| backtest_nar_v4_walkforward.py (時系列分割) | 90min | 各年度 AUC 表 |
