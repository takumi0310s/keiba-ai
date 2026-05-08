# Session #55 B: V20 expanding features 実装

**作成**: 2026-05-09 (Session #55 B、 dev/v20-expanding)
**file**: tools/v20_expanding_features.py
**output**: data/v20/expanding_features_v1.parquet

---

## 1. 実装

V18/V19 sib_w5 と同じ pattern (window 限定 expanding):
- group sort by (group_col, race_dt, race_id, umaban)
- shift(1) で当該行除外
- rolling(window) で直近 N 件
- Bayesian smoothing (alpha + prior) で低サンプル時の過学習防止

```python
def _expanding_rolling_rate(df, group_col, target_col, window, alpha, prior):
    rolled_sum = g.shift(1).rolling(window).sum()
    rolled_cnt = g.shift(1).rolling(window).count()
    rate = (rolled_sum + alpha * prior) / (rolled_cnt + alpha)
    return rate
```

---

## 2. 6 expanding features

| # | feature | group | target | window | alpha | prior | coverage |
|---|---------|-------|--------|--------|-------|-------|----------|
| 1 | jockey_wr_w30 | jockey_id | win | 30 | 5.0 | 0.10 | 100.0% |
| 2 | jockey_top3_w30 | jockey_id | top3 | 30 | 5.0 | 0.30 | 100.0% |
| 3 | trainer_top3_w90 | trainer_id | top3 | 90 | 10.0 | 0.30 | 100.0% |
| 4 | horse_career_top3_w5 | horse_id | top3 | 5 | 2.0 | 0.30 | 100.0% |
| 5 | horse_career_wr_w5 | horse_id | win | 5 | 2.0 | 0.10 | 100.0% |
| 6 | horse_career_top3_w10 | horse_id | top3 | 10 | 3.0 | 0.30 | 100.0% |

---

## 3. 統計

| feature | mean | std |
|---------|------|-----|
| jockey_wr_w30 | 0.0770 | 0.0585 |
| jockey_top3_w30 | 0.2310 | 0.1106 |
| trainer_top3_w90 | 0.2275 | 0.0762 |
| horse_career_top3_w5 | 0.2558 | 0.1723 |
| horse_career_wr_w5 | 0.0849 | 0.0918 |
| horse_career_top3_w10 | 0.2527 | 0.1471 |

→ 妥当 (理論値: win ~7.7%, top3 ~23%、 馬個別は分散大)

---

## 4. データ規模

- 期間: 2020-2025 (6 年)
- 総 rows: 283,714 (異常除外後)
- file size: 3.4 MB (parquet)
- coverage: 全 features 100% (Bayesian prior で 0 件レースも値が入る)

---

## 5. 5/9 V15 投資保護

✅ jra_races_full.csv は read-only
✅ V15 model 不変
✅ data/v20/ 新規 directory のみ書き込み

---

**Session #55 B 完了 (dev/v20-expanding)**
