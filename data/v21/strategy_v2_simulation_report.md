# Strategy Layer v2 - Backtest Simulation Report

**実施日**: 2026-05-16 12:31
**source**: `data/cumulative_results.csv` (settled rows)
**output CSV**: `data\v21\strategy_v2_simulation.csv`

## 1. 主要 metrics

| metric | baseline (cumulative 実 bet) | strategy_7 only (再現) | strategy_v2 full |
|--------|------------------------:|------------------------:|-----------------:|
| 対象 races | 529 | 529 | 529 |
| 戦略⑦ 除外 races | (mixed) | 63 | 63 |
| bet races | 529 | 466 | 466 |
| total inv | 370,300 | 326,200 | 385,700 |
| total pay | 345,230 | 316,080 | 339,940 |
| ROI | 93.23% | **96.90%** | 88.14% |
| PnL | -25,070 | -10,120 | -45,760 |
| delta vs baseline | - | +3.67pt | -5.09pt |

### bet_size 内訳 (v2)
- 700 円 (base): 419
- 1400 円 (2x):  9
- 2100 円 (3x):  38

### 戦略⑦ 除外 内訳
- 06_平場特別: 36
- 条件 E (頭数<=7): 11
- 条件 B (重〜不馬場): 16
- 距離<=1000m: 0

## 2. ★ data 制約 (★ 最重要 ★)

- `cumulative_results.csv` 総 settled rows = **529**
- そのうち **cumulative top1_score available rows = 20** (5/10 以前 score 未書込 既知 bug)
- `daily_predictions_full/` から enrich (top1_score + score_sum + odds): **70 races**
- 結果: strategy_v2 が **完全 EV 評価できた settled rows = 31**
- 残り rows は **score_unavailable_fallback** で 戦略⑦ pass+700 円 bet 動作 (= s7_only と同じ)

### 純粋 enrich subset 比較 (N=51)

| metric | baseline | strategy_v2 (real odds) | delta |
|--------|---------:|------------------------:|------:|
| inv | 35,700 | 95,200 | +59,500 |
| pay | 14,760 | 38,620 | +23,860 |
| ROI | 41.34% | 40.57% | -0.78pt |

bet_size 別 ROI (enrich subset only、 v2 view):

| bet_size | N | inv | pay | ROI |
|---------:|--:|----:|----:|----:|
| 700 円 | 4 | 2,800 | 680 | 24.3% |
| 1400 円 | 9 | 12,600 | 8,600 | 68.3% |
| 2100 円 | 38 | 79,800 | 29,340 | 36.8% |

### 観察 (honest)

- enrich subset で **v2 ROI 40.6% vs baseline 41.3% (delta -0.78pt)**
- sample 51 件は **5/10 1 日分のみ** で 5/10 自体が低調 day (baseline ROI 41.3%)
- 統計的 sample 不足 (n<100、 day-level 偏り 重大)
- 観測上 -0.8pt 悪化、 calibrator 過剰飽和の 影響 可能性

## 3. 結論 (honest)

### Strategy 7 only 部分 (valid signal)

- ROI 93.23% → 96.90% (delta +3.67pt)
- 戦略⑦ 除外 63 race で 投資効率 改善
- ★ ただし cumulative にも 既に 戦略⑦ 一部適用済 race が混ざっており、 純粋差分は 上記より控えめ

### EV 動的閾値部分 (sample 不足)

- v2 EV 評価できた sample 31 件 (= daily_predictions_full enrich 適用 後)
- sample 51 件 (enrich subset、 5/10 1 日のみ) で -0.8pt 悪化、 統計的に 未実証
- 「期待 +15-30% ROI 改善」 は **想定** 値、 5/18+ 蓄積 必要
- 残り 498 rows は score_unavailable で 戦略⑦ pass のみで base 動作

### Calibration 部分 (audit 別途)

- 学習 sample 21 件、 isotonic は p>=0.3 で 1.0 飽和
- blend 0.3 で慎重に取り込むが、 calibrator 自体の信頼性 低い
- 詳細: `data/v21/strategy_layer_calibrator_audit.md`

## 4. 次 step (真の検証)

- [ ] 5/18 (土) 朝から `tools/strategy_layer_v2.py --shadow YYYYMMDD` 起動
- [ ] 当日 odds_full.json と merge して 真の EV 計算 (shadow_eval 内で 拡張)
- [ ] 30+ race 蓄積後 `data/v21/strategy_v2_shadow_*.csv` を集計
- [ ] 200+ race 蓄積後 calibrator 再 train (現状 21 sample 不足)
- [ ] 真 ROI delta を honest に算出、 「+15-30%」 想定との 突合