# Phase 21B v2: V18 paper trade 再確認 + 5/17 ready (Session #87 Phase 21B 再実行)

> Session #87 (5/11) Phase 21B v2 — c8c1a2da の honest report 再検証
> ★ honest 厳守 ★ — fabricate しない、 データのみで判定

---

## 0. 再実行 trigger

Session #86 完了後、 Phase 21A (5/17 GO worksheet) commit 済。
本 v2 = c8c1a2da の Phase 21B を **同じ task で 再実行** + threshold scan + 5/17 dry run 追加。
結果は Phase 21B 初版と整合 (V18 model 状態 不変)。

---

## 1. 5/10 paper trade 再実行 (5-model)

`python tools/paper_trade_v22_runner.py --date 20260510` 実行 (schtask-friendly)。
loaded **34 races** (35R 想定 → 1R は backtest_engine load 対象外、 結果 csv 起源)。

| model | n_bet | n_hit | inv | pay | pnl | ROI | hit% |
|-------|-------|-------|------|------|-----|------|------|
| V15 (案B改 strict) | 6 | 2 | 4,200 | 3,490 | **-710** | **83.1%** | 33.3% |
| V18 cand | 1 | 1 | 700 | 680 | -20 | 97.1% | 100% |
| V20 cand | 1 | 1 | 1,000 | 971 | -29 | 97.1% | 100% |
| V21 cand (placeholder) | 1 | 1 | 1,000 | 971 | -29 | 97.1% | 100% |
| V22 RL (PPO 5000step) | 0 | 0 | 0 | 0 | +0 | 0% | 0% |

→ Phase 21B 初版 commit c8c1a2da と完全一致 ✅

---

## 2. ★ 重要発見 (5/10 詳細解析、 v2 で追加) ★

### 2.1 V15 が bet した 6 R 詳細

| race_id | course | R | score | cond | hit | payout | V18@0.75 通過? |
|---------|--------|---|-------|------|-----|--------|----------------|
| 202608030609 | 京都 | 9 | 0.716 | A | 0 | 0 | ❌ |
| 202608030611 | 京都 | 11 | **0.809** | C | **1** | 680 | ✅ |
| 202608030612 | 京都 | 12 | 0.734 | D | 0 | 0 | ❌ |
| 202605020605 | 東京 | 5 | 0.729 | C | 0 | 0 | ❌ |
| **202605020608** | **東京** | **8** | **0.723** | **D** | **1** | **2,810** | **❌ ← V18 miss!** |
| 202605020611 | 東京 | 11 | 0.703 | C | 0 | 0 | ❌ |

★ **V18 strict 0.75 filter が 東京 R8 hit (+2,810 payout) を miss** ★

東京 R8: score 0.723 → 0.75 未満で V18 除外 → 大当たり 1 件 取りこぼし。
V18 が hit した 京都 R11: score 0.809、 payout 680 (small)。

### 2.2 counterfactual: V18 が 東京 R8 を bet していた場合

- bet 数 2 / hit 数 2 / inv 1,400 / pay 3,490 / pnl **+2,090** / ROI **249%**

→ V18 strict 0.75 = 過剰 filter。 5/10 sample n=2 では 0.72 threshold が最適だった。

### 2.3 threshold scan (5/10 1日)

| threshold | bet | hit | ROI | pnl |
|-----------|-----|-----|-----|------|
| 0.60 | 13 | 5 | 66.6% | -3,040 |
| 0.65 | 9 | 4 | 73.5% | -1,670 |
| 0.70 | 6 | 2 | 83.1% | -710 |
| 0.75 | 1 | 1 | 97.1% | -20 |
| 0.80 | 1 | 1 | 97.1% | -20 |

→ 全 threshold で 1 日 net 負け (5/10 は荒れ気味)。 sample n=34 では 統計的有意性 なし。

★ honest: 1 日のみで V18 strict 採用 / 0.75 妥当性は **判定 不能** ★

---

## 3. ★ V18 model 不在 確認 (v2 で再検証) ★

```
$ ls models/v18/
(empty)
$ ls keiba_model_v18*.pkl.gz
(no match)
$ ls tools/predict_core_v18.py
predict_core_v18.py exists  ← 207 features 定義のみ
$ ls tools/train_v18_truevalue.py
train_v18_truevalue.py exists  ← 学習 script ready (5/16 user CLI 実行 base)
```

→ V18 candidate 207 features = **6 真値 (Phase 11) + 9 constant default + 192 V15 base**
→ V18 model 学習 = ★ **未実施** ★、 paper trade の bet decision は V15 score 由来のみ。

---

## 4. constant default 9 features 影響 (paper trade context)

| feature | constant value | V18 model 学習時 想定 importance |
|---------|---------------|--------------------------------|
| gaika_top3r_3r | 0.33 | ≒ 0 (variance=0) |
| gaika_winrate | 0.20 | ≒ 0 |
| gaika_dist_winrate | 0.20 | ≒ 0 |
| odds_change_3h_v18 | 0.0 | ≒ 0 |
| odds_change_30m_v18 | 0.0 | ≒ 0 |
| popularity_shift_v18 | 0 | ≒ 0 |
| odds_volatility_v18 | 0.0 | ≒ 0 |
| return_horse_score | 0.0 | ≒ 0 |
| saddle_room_score | 0.0 | ≒ 0 |

★ paper trade では **完全に影響なし** ★ — V18 model 不在 → bet decision に default features 関与せず。
★ V18 model 学習時 (5/12-5/16) は LGB tree が variance=0 で split 不可 → importance 0 想定 ★。

---

## 5. 真値化 6 features (Phase 11 commit 376f494f) 効果

| feature | data source | paper trade 効果 |
|---------|-------------|------------------|
| jockey_dist_winrate | KYI lookup | **未評価** (V18 model 不在) |
| jockey_track_winrate | KYI lookup | **未評価** |
| jockey_class_winrate | KYI lookup | **未評価** |
| jockey_x_trainer_wr | KYI x KKA | **未評価** |
| paddock_eval_v18 | KYI 18 cols | **未評価** |
| gaika_id_enc | KYI 放牧先 | **未評価** |

★ honest: **paper trade だけでは 真値化 6 features 効果 一切 評価 不能** ★
評価 path = V18 model 学習 → WF AUC 比較 (V15 0.8939 vs V18 ?) → feature_importance 確認。
予定 = 5/12-5/13 真値化 完了 + 5/16 user CLI で `python tools/train_v18_truevalue.py` 実行。

---

## 6. 5/17 paper trade ready (schtask 化)

### 6.1 dry run 確認 (5/17 今日時点 race 0 件)

```
$ python tools/paper_trade_v22_runner.py --date 20260517
[paper_v22_runner] loaded 0 races
[paper_v22_runner] no races to evaluate
rc=0 (schtask-friendly)
```

→ ✅ exception なし、 race 0 件でも safe exit、 V15 production を巻き込まない。

### 6.2 schtask 候補

| 名 | 時刻 | command |
|----|------|---------|
| PaperTradeV22_Daily | 20:00 (daily_results 後) | `python tools/paper_trade_v22_runner.py --date %DATE% --notify` |

### 6.3 5/17 並行運用 chain

1. ✅ 5/17 朝 V15 本番 daily_predict (既 schtask)
2. ✅ 5/17 投票 (V15 案B改 strict)
3. ✅ 5/17 夜 daily_results.py
4. ✅ 5/17 夜 paper_trade_v22_runner --date 20260517 --notify (V15/V18/V20/V21/V22 5-model 比較)
5. ✅ Discord #updates へ summary 通知 (dedup 経由)

### 6.4 V18 schtask 上の constant default 表示

paper_v18 = V15 score + 0.75 filter のため V18 model 不要、 schtask で warning 不要。
V18 model 学習後 (5/16+) は別途 V18 production strategy 実装 + schtask 化。

---

## 7. V15 投資保護 (絶対遵守 確認)

| 項目 | 状態 |
|------|------|
| tools/predict_core.py | ✅ 不変 |
| keiba_model_v15_central_live.pkl.gz | ✅ 不変 |
| daily_predict.py 動作 | ✅ 既 schtask、 paper trade と分離 |
| paper_trade_v22 = read-only | ✅ V15 model load only |
| 累計 +¥14,140 | ✅ 維持 |

---

## 8. ★ honest summary v2 ★

| 項目 | 結果 |
|------|------|
| paper trade 動作 | ✅ 確認 (34R load, 5-model 評価) |
| V18 model file 存在 | ❌ **なし** (Phase 19 学習 未実施) |
| V18 paper strategy 中身 | V15 score + 0.75 filter のみ |
| 真値化 6 features 効果 (paper trade) | ★ **完全 未評価** ★ |
| constant default 9 features 影響 (paper trade) | ★ **完全になし** ★ (V18 model 不在) |
| 5/10 V18 大当たり miss | ★ **東京 R8 (0.723 score) +2,810 payout** miss ★ |
| 5/17 schtask ready | ✅ rc=0、 exception 安全、 V15 巻き込まず |
| V15 投資保護 | ✅ 完全 |

---

## 9. 次 step (5/12+)

1. 5/12-5/13: Phase 11 残 9 features 真値化 (gaika 外厩 + odds_change + return_horse)
2. 5/14-5/15: Phase 12/13 真値化 (JV-Link DataLab + netkeiba)
3. 5/16: `python tools/train_v18_truevalue.py` user CLI 実行 → V18 model 完成
4. 5/16-5/17: V18 WF AUC 評価 (vs V15 0.8939)、 真値化 6 features feature_importance 確認
5. 5/17: V18 並行 paper trade (schtask 化)
6. 5/24+: Phase 3 V20 構築 移行

---

## 10. 結論

✅ **A1**: 5/10 paper trade 再実行 (V15 ROI 83.1% / V18 cand 97.1% / V20 97.1% / V21 97.1% / V22 0%)
✅ **A2**: V18 cand n=1 hit、 sample size 不足、 V18 strict 0.75 が **東京 R8 +2,810 大当たり miss** 確認
✅ **A3**: constant default 9 features 影響 = paper trade では **完全になし** (V18 model 不在)
✅ **B1**: 5/10 score 帯別 threshold scan (0.60/0.65/0.70/0.75/0.80 全敗、 sample 1 日 不足)
✅ **B2**: 真値化 6 features 効果 = ★ **paper trade では 評価不能、 V18 学習 + WF AUC 必須** ★
✅ **C1**: 5/17 schtask dry run rc=0、 exception 安全、 V15 巻き込まず
✅ **C2**: constant default warning は V18 model 不在で paper trade 影響なし、 学習時のみ警告
✅ **V15 投資保護 完全**
