# 5/21 8 並列投入 完了レポート
> 完了: 2026-05-21 / V15 production 完全不変

---

## 全 task 完了サマリー

| task | commit | 状態 | key finding |
|------|--------|------|-------------|
| [a] data 補填強化 | `f0e32895` | ✅ | data 100% complete / 5/9 二重 append 33件 (ROI 真値は重複除去済み正) / paper eval base n=596 確定 |
| [b] C4 param tune | `be3e3920` | ✅ | skip 1600-2000m + 6pt → hold-out ROI **178.1%** (現行 overfit 47.8pt) / n=16 で paper 継続必須 |
| [c] hook verify | `b8f08a89` | ✅ | ★★ **8 strategy paper が全 None** のバグ発見 → [c-fix] で即修正 |
| [c-fix] race_auto_notify | `4d54ccf4` | ✅ | **5/23 SAT から 8 strategy 真の蓄積開始** (predictions 渡し修正) |
| [d] top-5 score weight | `2eefcc97` | ✅ | tool 完成 / top2_score 欠損多く 5/23+ データ蓄積後に再 backtest |
| [e] ensemble blending | `81c1eb5a` | ✅ | tool 完成 / V15 single degraded mode (candidates 不在、 parity 確認済) |
| [f] confidence score | `a1c687bf` | ✅ | ★最推奨★ tool 完成 / num_horses 欠損でフィルタ効かず → 5/23+ 蓄積後に再実行 |
| [g] divergence v2 | `2c4d0546` | ✅ | tool 完成 / pop_rank 欠損で proxy mode → race_notify_log v2 蓄積後に再実行 |
| [電源 案 C] | `b96c06a4` | ✅ | **★ 完全可能 ★** / JRDB KYI 火〜木 = 元々空データ / schtask 変更不要 |

---

## 重大発見: [c-fix] — 8 strategy paper 全 None バグ

### 問題
`race_auto_notify.py` の `_v2_log_phase2_safe()` が `predictions=None` で呼ばれており、
`race_notify_log_v2` の `build_strategy_formations()` に predictions が渡らず、
**8 strategy (actual/c3/c4/no_1pop/divergence/ev_filter/odds_filter) が全て None** だった。

→ race_notify_log v2 の paper 蓄積が実装以来ゼロだった可能性。

### 修正 (既存 logic 完全不変)
`_v2_log_phase2_safe()` の call site で df から predictions list を構築:
```python
_preds_for_v2 = [{'horse_num': int(row['馬番']), 'pop_rank': int(row.get('pop_rank', 99)), 'odds': float(odds_dict.get(uma, 99.0))} for _, row in df.iterrows()]
```
→ **5/23 SAT から 8 strategy 真の paper 蓄積が開始する。**

---

## [b] C4 param tune 発見

| 設定 | skip 距離 | tickets | Train ROI | Hold-out ROI | overfit |
|------|---------|---------|-----------|-------------|---------|
| 現行 | 1600-1800 | 7 | 142.6% | 94.8% | ✅ True (+47.8pt) |
| **最良** | **1600-2000** | **6** | 185.4% | **178.1%** | False |

★ n_test=16 のため統計的確信度は低い。paper 評価継続で 6/17 判定。

---

## [電源 案 C] 完全可能

### verdict: ★ 完全可能 ★

- JRDB KYI: 火〜木は元々空データ → miss してもゼロ影響
- JRA premium scrape: 平日中央開催なし → 影響なし
- NAR: shadow 評価のみ、実投票なし → 影響軽微 (Strategy 8 go/no-go を 7/1 に延期で対処)
- schtask 変更不要

### 実施方法
1. 火曜夜にシャットダウン
2. 金曜朝 06:00 前後に起動
3. それだけ

### 節約
- vs 完全 ON: 月約 464 円 / 年約 5,616 円
- vs スリープ (案 B): 追加月 47 円のみ

### 唯一の注意
- NAR Strategy 8 サンプルが週 3 日減少 → 6/15 go/no-go を 7/1 Phase 3 統合評価に延期

---

## paper 蓄積状況と再実行タイミング

| tool | 現状 | 再 backtest タイミング |
|------|------|--------------------|
| [b] C4 param | hold-out n=16 小 | 5/23-6/7 で n=100+ 蓄積後 |
| [d] top-5 score weight | top2_score 欠損 | 5/23+ `daily_predict.py` が top2_score 書き込み開始後 |
| [f] confidence score | num_horses 欠損 | 5/23+ num_horses 蓄積後 |
| [g] divergence v2 | pop_rank 欠損 | race_notify_log v2 蓄積後 (pop_rank 記録開始) |

---

## 5/23 SAT 完成形

```
06:00 起床 (金曜夜から継続 or 土曜朝起動)
08:30 LiveOrchestrator 真の運用開始 (mock=True)

実 cash: V15 + 戦略⑦案 C + C4 (production 完全不変)

paper 並行蓄積 (race_notify_log v2 phase 2 — 初めて真のデータ):
  actual / c3 / c4 / c3c4 / no_1pop / divergence / ev_filter / odds_filter

新 paper tools (5/23 から手動呼び出し可能):
  tools/c4_param_tune.py         [b] C4 grid
  tools/v15_top5_score_weight.py [d] score weighting
  tools/v15_ensemble_blender.py  [e] ensemble
  tools/race_confidence_score.py [f] confidence ★最推奨★
  tools/v15_market_divergence_v2.py [g] divergence v2
```

---

## 次アクション

| 優先 | アクション | タイミング |
|------|---------|---------|
| ★ | 電源 案 C 実施 (火曜夜 OFF → 金曜朝 ON) | 5/22 (木) 夜から試行 |
| ★ | 5/23 SAT race_notify_log v2 の 8 strategy が None でないことを確認 | 5/23 レース後 |
| 中 | [f] confidence score 再 backtest (num_horses 蓄積後) | 6/1+ |
| 中 | [b] C4 hold-out 追加サンプル蓄積 | 5/31 (n=50+) で中間確認 |
| 低 | cumulative_results.csv の 5/9 二重 append 修正 | 任意 (ROI 計算は重複除去済みで正) |
