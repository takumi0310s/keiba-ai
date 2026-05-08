# Sprint 2 A: horse_weight_features (Session #47 A)

**作成**: 2026-05-08 (Session #47 A、 dev/sprint2)
**目的**: 直近 3 走の体重 trend + 変化率 + 同条件比較

---

## 1. features (4 件)

| feature | type | 計算 |
|---------|------|------|
| weight_trend_3r | int (-1/0/+1) | 直近 3 走 体重差 -4kg/+4kg threshold |
| weight_change_pct_3r | float | 連続変化率 mean (%) |
| weight_vs_same_cond | float (kg) | 同 course/distance 過去平均との差 |
| weight_extreme_change_3r_count | int | ±10kg 超 変化 過去 3 走 count |

---

## 2. backtest (jra_races_full.csv 532K rows)

```
n: 531,619 races
horse_weight_std corr target: +0.0493 (weak positive)
interpretation: 体重 std 高 = 一時的に top3 多 (bias 含む)
```

→ feature として weak だが、 trend + extreme + same_cond 組み合わせで AUC +0.001-0.003 期待

---

## 3. production 統合 plan (5/15 merge 後)

```python
from tools.horse_weight_features import compute_horse_weight_features
feats = compute_horse_weight_features(history_df, horse_id, course, distance)
# → race_df に 4 columns 追加
```

V20 構築時 (Phase 3 後半 5/16-6/8) に既存 features 150 + Sprint 2 各 idea で AUC +0.005-0.01 累積期待

---

## 4. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**

---

**Session #47 A 完了 (dev/sprint2)**
