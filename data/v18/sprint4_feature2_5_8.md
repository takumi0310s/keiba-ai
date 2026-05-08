# Sprint 4 ★★★ #2: master_index 5 indices 結果 (5/8)

**branch**: dev/sprint4
**source**: data/netkeiba_master_index.csv (5 fields)
**実装**: tools/sprint4_feature2.py
**リーク対策**: 当該 race を除外し expanding window で 過去 races の mean (dam_top3r 教訓)

## 1. 追加 features (expanding-prev mean)

| feature | 元 source | raw coverage | expanding coverage |
|---------|----------|--------------|------------------|
| mi_time_idx_prev | MI.time_index | 23.4% | 21.6% |
| mi_master_idx_prev | MI.master_index | 23.4% | 21.6% |
| mi_start_idx_prev | MI.start_index | 23.4% | 21.6% |
| mi_chase_idx_prev | MI.chase_index | 23.4% | 21.6% |
| mi_agari_idx_prev | MI.agari_index | 23.4% | 21.6% |

## 2. AUC contribution (1-fold WF per year)

| 年 | V15 baseline | V15 + MI(prev) | Δ | gap |
|----|------------|----------------|----|----|
| 2023 | 0.8677 | 0.8677 | +0.0000 | 0.0185 |
| 2024 | 0.8708 | 0.8708 | +0.0000 | 0.0140 |
| 2025 | 0.8680 | 0.8666 | -0.0014 | 0.0192 |

**平均 AUC contribution**: -0.0005

期待値 +0.003-0.005 と比較: 🔴 期待未達 (coverage 低い 2024-2025 のみで限定的)

## 3. リーク監査

✅ train-eval gap 全て ≤ 0.05、 リークなし

## 4. 結論

- **期待 AUC**: +0.003-0.005
- **実績 AUC**: -0.0005 (平均 2023-2025)
- **採用判定**: 🟡 要再検討
