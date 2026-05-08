# Sprint 4 V15.5 PoC: V15 + ★★★ 13 features 統合 (5/8)

**branch**: dev/sprint4
**期待**: V15 0.8788 → V15.5 0.894-0.899
**実装**: tools/v15_5_features.py + tools/sprint4_v15_5_poc.py

## 1. V15.5 構成

- V15 base: 145 features
- ★★★ 追加: 13 features
- V15.5 合計: 158 features

### 追加 13 features

| # | feature | source | coverage |
|---|---------|--------|----------|
| 1 | srb_bias_1c | JRDB SRB | 22.4% |
| 2 | srb_bias_2c | JRDB SRB | 24.4% |
| 3 | srb_bias_bs | JRDB SRB | 53.2% |
| 4 | srb_bias_3c | JRDB SRB | 53.3% |
| 5 | srb_bias_4c | JRDB SRB | 53.3% |
| 6 | srb_bias_st | JRDB SRB | 53.7% |
| 7 | mi_time_idx_prev | netkeiba MI | 21.6% |
| 8 | mi_master_idx_prev | netkeiba MI | 21.6% |
| 9 | mi_start_idx_prev | netkeiba MI | 21.6% |
| 10 | mi_chase_idx_prev | netkeiba MI | 21.6% |
| 11 | mi_agari_idx_prev | netkeiba MI | 21.6% |
| 12 | jo_cid_idx | JRDB JO | 55.2% |
| 13 | jo_ls_idx | JRDB JO | 55.2% |

## 2. AUC: V15 vs V15.5

| 年 | V15 | V15.5 | Δ | gap (train-eval) |
|----|-----|------|----|-----------------|
| 2023 | 0.8677 | 0.8677 | -0.0000 | 0.0188 |
| 2024 | 0.8708 | 0.8709 | +0.0002 | 0.0137 |
| 2025 | 0.8680 | 0.8668 | -0.0012 | 0.0190 |
| **平均** | **0.8688** | **0.8685** | **-0.0003** | — |

## 3. 期待値比較

- 期待 V15.5 AUC: 0.894-0.899
- 実績 V15.5 AUC: 0.8685
- 期待 contribution: +0.008-0.013
- 実績 contribution: -0.0003

🔴 マイナス寄与 (V15.5 統合 NO-GO、 個別 feature 再検討)

## 4. クラス別 AUC (eval 平均 2023-2025)

| 条件 | V15 | V15.5 | Δ |
|------|-----|------|----|
| A | 0.8615 | 0.8609 | -0.0005 |
| B | 0.8513 | 0.8506 | -0.0007 |
| C | 0.8767 | 0.8764 | -0.0003 |
| D | 0.8640 | 0.8637 | -0.0003 |
| E | 0.8621 | 0.8609 | -0.0011 |
| X | 0.8674 | 0.8688 | +0.0014 |

## 5. リーク監査

✅ train-eval gap 全て ≤ 0.05、 リークなし

## 6. 結論

🔴 V15.5 統合 期待未達。 個別 feature 採用判定が必要。

## 7. 投資保護 確認

- main branch: 6c0680ad (不変)
- V15 model file: 不変 (keiba_model_v135_*.pkl.gz)
- predict_core / daily_predict / app.py: 不変
- schtasks 41 件: 不変
- 5/9 朝 V15 daily_predict 動作: 完全同一保証 ✅
