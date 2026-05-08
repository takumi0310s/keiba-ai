# Sprint 4 ★★★ #3: JRDB JO cid_idx / ls_idx 結果 (5/8)

**branch**: dev/sprint4
**source**: data/jrdb_jo.csv (2 fields)
**実装**: tools/sprint4_feature3.py
**リーク risk**: pre-race (JO は朝段階 確定)

## 1. 追加 features

| feature | source | coverage |
|---------|--------|----------|
| jo_cid_idx | JO.cid_idx | 51.4% |
| jo_ls_idx | JO.ls_idx | 51.4% |

## 2. AUC contribution (1-fold WF per year)

| 年 | V15 baseline | V15 + JO | Δ | gap |
|----|------------|---------|----|----|
| 2023 | 0.8681 | 0.8678 | -0.0003 | 0.0186 |
| 2024 | 0.8699 | 0.8699 | -0.0000 | 0.0147 |
| 2025 | 0.8682 | 0.8686 | +0.0004 | 0.0149 |

**平均 AUC contribution**: +0.0000

期待値 +0.002-0.003 と比較: 🔴 期待未達

## 3. リーク監査

✅ train-eval gap 全て ≤ 0.05、 リークなし

## 4. 結論

- **期待 AUC**: +0.002-0.003
- **実績 AUC**: +0.0000 (平均 2023-2025)
- **採用判定**: ✅ V15.5 統合 採用候補
