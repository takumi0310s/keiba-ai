# Phase 21B: V18 paper trade 動作確認 + V15 retrain 同等性 honest report

> Session #87 Phase 21B (5/10 22:30+)
> ★ honest report 厳守 ★ — fabricate しない

---

## 1. 5/10 35 R paper trade 実結果

`tools/paper_trade_v22_runner.py --date 20260510` 実行 (5-model 並行):

| model | bet | hit | inv | pay | pnl | ROI | hit% |
|-------|-----|-----|------|------|-----|------|------|
| V15 (案B改 strict) | 6 | 2 | 4,200 | 3,490 | **-710** | **83.1%** | 33.3% |
| V18 cand | 1 | 1 | 700 | 680 | -20 | 97.1% | 100% |
| V20 cand | 1 | 1 | 1,000 | 971 | -29 | 97.1% | 100% |
| V21 cand (placeholder) | 1 | 1 | 1,000 | 971 | -29 | 97.1% | 100% |
| V22 RL (PPO 5000 step) | 0 | 0 | 0 | 0 | +0 | 0% | 0% |

出力: `data/v22/paper_trade_summary_20260510.csv` + `paper_trade_detail_20260510.csv`

---

## 2. ★ honest 重要発見 (V18 ≈ V15 retrain 同等性) ★

### 2.1 V18 cand strategy logic 確認

`tools/paper_trade_engine_v22.py:37-44` より:

```python
def paper_v18(race: RaceRecord) -> dict:
    """V18 candidate: V15 + より厳しい score 閾値 (0.75+ 限定)"""
    base = paper_v15(race)
    if base['bet'] == 0:
        return base
    if race.morning_top1_score < 0.75:
        return {'bet': 0, 'reason': 'V18_strict_score'}
    return {'bet': 700, 'reason': 'V18_strict'}
```

### 2.2 ★ V18 model file 不在 確認 ★

```
$ ls models/v18/
(no such directory)
$ ls keiba_model_v18*.pkl.gz
(no match)
```

★ V18 model file は **存在しない** ★

### 2.3 V18 paper trade の意味

- ★ V18 cand strategy は V15 score を **より厳しく filter している だけ** ★ (0.7 → 0.75)
- V18 で生成される score は **V15 model の score** (V18 model 未学習)
- predict_core_v18.py で 207 features 定義あるが、 model 未学習 = features は使われていない
- ★ 真値化 6 features (Phase 11) / default 9 features は paper trade に **影響なし** ★ (V18 model が無いため)

### 2.4 V18 ≈ V15 retrain ?

「V18 ≈ V15 retrain」 = 「V18 が V15 と同じ score を返す」 という意味なら:
- ✅ 現状確認: V18 paper trade = V15 score ベース。 score 値 完全同一
- ❌ 真の V18 retrain (separate model file with 207 features) = ★ 未学習、 まだ存在しない ★

### 2.5 5/10 1 R hit の解釈

V18 が hit した 1 R (京都 R11 平城京 S OP特別、 score=0.809):
- これは V15 score 0.809 を 0.75 filter 通過した R
- V18 model が無いので、 真値化 6 features の効果ではない
- 単純に V15 で score 高 R が trio hit した偶然 (sample size n=1)

→ ★ 真値化 6 features の効果は V18 model 学習 + WF 評価で初めて評価可能 ★ (Phase 19 で plan、 5/16 user CLI ready)

---

## 3. constant default 9 features 影響 確認

### 3.1 paper_trade_v22 では 影響なし
- V18 model 不在 → score 算出に default features 関与せず
- bet decision = V15 score のみ

### 3.2 V18 model 学習時 (Phase 19 完了後) の予想
- 真値化 6 features: feature_importance > 0 期待 (KYI 騎手 winrate 等)
- default 9 features: feature_importance ≒ 0 想定 (constant value のため LGB が無視)
  - gaika_top3r_3r = 0.33 (constant)
  - gaika_winrate = 0.20 (constant)
  - gaika_dist_winrate = 0.20 (constant)
  - odds_change_3h_v18 = 0.0 (constant)
  - odds_change_30m_v18 = 0.0 (constant)
  - popularity_shift_v18 = 0 (constant)
  - odds_volatility_v18 = 0.0 (constant)
  - return_horse_score = 0.0 (constant)
  - saddle_room_score = 0.0 (constant)

★ 9 default 特徴量は constant variance = 0、 LGB tree split に使用不可 → feature_importance = 0 想定 ★

### 3.3 V18 retrain 同等性 (predicted)

V18 = V15 + 6 真値化 + 9 constant default features

期待:
- 6 真値化が無視できる小寄与 → V18 ≈ V15 (差分 < 0.01 AUC)
- 6 真値化が小寄与あり → V18 = V15 + 0.001-0.005 AUC
- LGB tree が 9 default feature 完全無視 → V18 ≈ V15 retrain (training noise 差分のみ)

→ ★ 同等性検証は Phase 19 V18 model 学習後 に WF AUC で確定 ★

---

## 4. 5/10 score 帯別 比較 (V15 案 B 改 + V18 strict)

V15 ベットした 6 R 詳細 (paper_trade_detail から):

| race_id | course | R | top1_score | trio_hit | V15 bet | V18 bet 通過? |
|---------|--------|---|-----------|----------|---------|---------------|
| 202608030611 | 京都 | R11 | 0.809 | 1 | ✅ 700 | ✅ 700 |
| 202604010406 (推定) | 新潟 | R6 | 0.7-0.8 | (要照合) | ✅ 700 | ❌ 0.75 未満 |
| 202604010407 | 新潟 | R7 | 0.7-0.8 | (要照合) | ✅ 700 | ❌ 0.75 未満 |
| 202604010411 | 新潟 | R11 | 0.7-0.8 | (要照合) | ✅ 700 | ❌ 0.75 未満 |
| 202605020607 | 東京 | R7 | 0.7-0.8 | (要照合) | ✅ 700 | ❌ 0.75 未満 |
| 202605020611 | 東京 | R11 | 0.703 (NHK マイル) | 0 | ✅ 700 | ❌ 0.75 未満 |

V15 hit 2 R / 6 bet → V18 strict は score 0.75+ 1 R に絞った結果 偶然 hit。

★ honest: V18 strict の改善は sample n=1 で **統計的有意性 なし** ★

---

## 5. 5/17 paper trade ready 確認

### 5.1 v22 runner schtask-friendly
- `--date YYYYMMDD` で 1 日のみ評価
- exception 時 exit 0 (V15 production を巻き込まない、 paper_trade_v22_runner.py:13)
- V15 model file は read-only

### 5.2 5/17 並行運用
- ✅ 5/17 朝 V15 本番 daily_predict 実行 (既 schtask)
- ✅ 5/17 夜 daily_results.py 完了後、 paper_trade_v22_runner.py --date 20260517 即実行可
- ✅ V15 投票 + V18/V20/V21 paper 比較 → Discord 通知
- ✅ V22 RL も 0 bet / 慎重すぎなら 学習 epoch 増加 後 再投入

### 5.3 schtask 候補 (5/17+)
```
名: PaperTradeV22_Daily
時刻: 20:00 (daily_results 完了後)
command: python tools/paper_trade_v22_runner.py --date <today> --notify
```

---

## 6. ★ honest summary ★

| 項目 | 結果 |
|------|------|
| V18 paper trade 動作 | ✅ 確認 (paper_trade_v22_runner.py --date 20260510) |
| V18 model file 存在 | ❌ **なし** (predict_core_v18.py 207 features 定義のみ) |
| V18 paper trade の中身 | V15 score + 0.75 filter (V15 score を厳しく絞っただけ) |
| 真値化 6 features 効果 | ★ **未評価**、 V18 model 不在のため ★ |
| 同等性検証 (V18 ≈ V15 retrain) | ★ **未確定**、 Phase 19 V18 model 学習後 WF AUC で判定 ★ |
| 5/10 1 hit の意味 | sample n=1、 score 0.809 → trio hit (V15 score 効果、 V18 真値化 寄与不明) |
| 5/17 paper trade ready | ✅ schtask-friendly、 並行運用 OK |

---

## 7. 次 step (Phase 19 完了後)

1. ★ V18 model 学習 ★ (tools/train_v18_real.py、 Phase 19 で ready)
2. WF AUC 評価 (V15 0.8939 vs V18 ?)
3. 真値化 6 features の feature_importance 確認
4. V18 ≈ V15 retrain 確定
5. 5/24+ で 残 9 features 真値化 → V18 v2 学習

---

## 8. V15 投資保護

✅ tools/predict_core.py / V15 model 不変
✅ paper_trade_v22 = read-only、 V15 model は load only
✅ 累計 +¥14,140 維持

---

## 9. 結論

✅ A1: 5/10 paper trade 実行 (V15 ROI 83.1% / V18 cand 97.1% / V20 97.1% / V21 97.1% / V22 0%)
✅ B1: V18 model 不在確認、 paper_v18 = V15 score + 0.75 filter
✅ B2: 真値化 6 features 効果 = ★ **未評価** ★ (V18 model 学習待ち、 Phase 19)
✅ B3: 同等性検証 = ★ **未確定** ★ (WF AUC 比較必要)
✅ C1: 5/17 paper trade ready 確認 (schtask-friendly)
✅ V15 完全保護
