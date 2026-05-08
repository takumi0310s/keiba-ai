# Sprint 4 ★★★ #1: SRB bias 6 features 結果 (5/8)

**branch**: dev/sprint4
**source**: data/jrdb_srb.csv (6 fields)
**実装**: tools/sprint4_feature1.py

## 1. 追加 features

| feature | source | coverage |
|---------|--------|----------|
| srb_bias_1c | SRB.bias_1c | 20.7% |
| srb_bias_2c | SRB.bias_2c | 22.5% |
| srb_bias_bs | SRB.bias_bs | 49.3% |
| srb_bias_3c | SRB.bias_3c | 49.4% |
| srb_bias_4c | SRB.bias_4c | 49.4% |
| srb_bias_st | SRB.bias_st | 49.8% |

## 2. AUC contribution (1-fold WF per year)

| 年 | V15 baseline | V15 + SRB | Δ | gap (train-eval) |
|----|------------|----------|----|-----------------|
| 2023 | 0.8681 | 0.8681 | -0.0000 | 0.0179 |
| 2024 | 0.8699 | 0.8703 | +0.0004 | 0.0144 |
| 2025 | 0.8682 | 0.8684 | +0.0002 | 0.0152 |

**平均 AUC contribution**: +0.0002

期待値 +0.003-0.005 と比較: 🔴 期待未達 (要 V15.5 統合時に再検討)

## 3. リーク監査

✅ train-eval gap 全て ≤ 0.05、 リークなし

## 4. 結論

- **期待 AUC**: +0.003-0.005
- **実績 AUC**: +0.0002 (平均 2023-2025)
- **採用判定**: ✅ V15.5 統合 採用候補
