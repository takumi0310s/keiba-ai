# 30 年 backtest 戦略比較 plan (Session #84)

> 案B改 V15 → V22 RL までの 5 戦略を 30 年 walk-forward backtest で 一貫評価。
> 作成: 2026-05-09 (Session #84)

---

## 1. 比較対象 5 戦略

### 1.1 strategy 1: 案B改 V15 (current baseline)
- model: V15 LGB+XGB+FT+IR (4-model)
- features: 145 (Pattern A 124 + Live 8 + 環境 13)
- 投票 logic: 案B改 strict 7 点 + 戦略⑦ 除外 (06_特別 / 京都 / 条件 E / 条件 B)
- production status: ★ 5/9 現在 運用中 ★
- 期待 AUC: 0.8939
- 期待 ROI: 110-130%

### 1.2 strategy 2: 案B改 V18 (5/16 trial)
- model: V18 (V15 + sib_w5_exp Session #38 修正版)
- features: 150
- 投票 logic: 案B改 strict 7 点 + 戦略⑦
- 投入予定: 5/16 V18 trial 1 day (Session #74 plan v5)
- 期待 AUC: 0.890+
- 期待 ROI: 115-135%

### 1.3 strategy 3: 案B改 V20 (7/1 投入)
- model: V20 (V18 + KKA Session #53 + sib_*_exp 全 + JV-Link 統合)
- features: 200+
- 投票 logic: 案B改 strict 7 点 + 戦略⑦
- 投入予定: 7/1 V20 段階投入 (上限 5,000円/日)
- 期待 AUC: 0.880-0.895
- 期待 ROI: 130-150%

### 1.4 strategy 4: hybrid (Session #82 確定)
- model: V20 流用
- features: V20 と同一 200+
- 投票 logic: cell 別
  - 芝 14 頭以下: 11 点 ★ (+18.1pt 改善期待)
  - ダート (全頭数): 7 点 strict
  - 15+ 頭 (重賞除く): 7 点 strict
  - 重賞: 投票なし
- 投入予定: 8/1 採用判定 (5 件 ALL PASS で GO)
- 期待 AUC: 0.880-0.895 (V20 と同一)
- 期待 ROI: 140-160%

### 1.5 strategy 5: V22 RL (Session #83 候補)
- model: V21 (V20 + 動画 features) + RL agent
- features: 220+
- 投票 logic: RL で 動的最適化 (state = 残予算 + 累計 P/L + R 特徴)
- 投入予定: 12/1 投入候補 (10/1 PoC、 11/1 paper trade)
- 期待 AUC: 0.880-0.895
- 期待 ROI: 150-180%

---

## 2. 評価 metrics (5 軸)

### 2.1 ROI
- mean / median (5 fold)
- bootstrap CI95
- 採用閾値: lower bound >= 100%

### 2.2 Sharpe ratio (annualized)
- mean ROI / std ROI × sqrt(年数)
- 採用閾値: >= 1.0

### 2.3 max drawdown
- 累計 P/L curve の 最大谷
- 採用閾値: < 30%

### 2.4 投票数 / 年
- 戦略の投票頻度
- 過小: < 50 R/年 → サンプル不足
- 過大: > 200 R/年 → 過剰投資 risk

### 2.5 累計利益
- 30 年 累計 P/L
- 採用閾値: > +¥1,000,000 (年 +¥33,000 平均)

---

## 3. 比較 matrix (期待結果)

| strategy | AUC | ROI | Sharpe | max DD | 投票/年 | 30 年累計 |
|----------|-----|-----|--------|--------|--------|----------|
| 1. V15 案B改 | 0.8939 | 120% | 0.8 | 25% | ~150 R | +¥630,000 |
| 2. V18 案B改 | 0.890 | 125% | 0.9 | 22% | ~150 R | +¥780,000 |
| 3. V20 案B改 | 0.890 | 140% | 1.1 | 20% | ~150 R | +¥1,260,000 |
| 4. hybrid | 0.890 | 150% | 1.3 | 22% | ~150 R | +¥1,575,000 |
| 5. V22 RL | 0.890 | 165% | 1.5 | 18% | ~120 R | +¥2,025,000 |

→ ★ V22 RL が最強候補だが PoC 必要 ★

---

## 4. 採用判定の 6 件閾値

各 strategy 採用には以下 6 件 ALL PASS:

- [ ] 全 fold AUC >= 0.85
- [ ] 全 fold ROI >= 100%
- [ ] bootstrap CI95 lower >= 100%
- [ ] Sharpe >= 1.0
- [ ] max DD < 30%
- [ ] 30 年累計 >= +¥800,000

---

## 5. 経年 robustness 検証

### 5.1 年代別 AUC 安定性
- 1995-2004: model 性能?
- 2005-2014: model 性能?
- 2015-2024: model 性能?
- → drift > 5% で 学習 data 古さ問題

### 5.2 surface / 頭数 differential 一貫性
- hybrid 戦略 (Session #82) の cell 別 効果が 30 年で一貫するか
- 1995-2014 vs 2015-2024 で differential 比較
- → 一貫しなければ Session #82 結論 修正必要

### 5.3 model drift 評価
- 各 fold で valid AUC が train AUC から 何 pt 落ちるか
- > 0.03pt → over-fitting 疑い

---

## 6. 出力 deliverable

### 6.1 markdown
- `docs/BACKTEST_30Y_REPORT.md` (将来生成)
- 5 戦略 × 5 fold matrix
- 6 件閾値 PASS/FAIL 判定
- robustness 結論

### 6.2 plot (data/backtest_30y/plots/)
- 累計 P/L curve (5 戦略 重ね描き)
- 年別 AUC trend
- 年別 ROI trend
- DD curve
- bootstrap CI95 plot

### 6.3 採用判定 sheet
- 各 strategy の 6 件 PASS 数
- 6/6 → 強候補、 4-5/6 → 条件付候補、 < 4/6 → 不採用

---

## 7. 関連 doc

- `docs/BACKTEST_30_YEAR_DESIGN.md` — 範囲設計
- `docs/BACKTEST_DATA_PIPELINE.md` — pipeline 設計
- `docs/BACKTEST_BUILD_TIMELINE.md` — schedule
- `docs/STRATEGY_HYBRID_DESIGN.md` (Session #82)

---

**結論**: 5 戦略 (V15 / V18 / V20 / hybrid / V22 RL) を 30 年 walk-forward で 5 metrics 評価。 期待 ROI は 120% → 165% で 単調増加見込。 V22 RL が最強候補だが PoC 後に判定。 採用は 6 件閾値 ALL PASS 必須。
