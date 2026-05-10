# Phase 15 — V20 個別モデル学習結果

**date**: 2026-05-10 21:00 (JST)
**run**: train/train_v20_ensemble.py --quick (LGB+XGB) + --with-ft (FT-Transformer)
**GPU**: RTX 4070 Ti SUPER 16GB

## 設定

| 項目 | 値 |
|------|----|
| WF split | train 2022-2024 / val 2025 |
| train rows | 140,867 |
| val rows | 47,497 |
| features | 145 (V15 base 実 signal のみ) |
| objective | binary (target=top3) |

## 結果

### LightGBM (GPU)

| metric | 値 |
|--------|----|
| device | gpu (lightgbm GPU build) |
| num_round | 600 (early_stop 40) |
| best_iter | ~300 |
| **val AUC** | **0.8662** |
| 学習時間 | 16.3 sec |

進行 (val AUC):
- iter 100: 0.8620
- iter 200: 0.8658
- iter 300: 0.8661

### XGBoost (GPU)

| metric | 値 |
|--------|----|
| device | cuda (tree_method=hist) |
| num_round | 600 (early_stop 40) |
| best_iter | ~440 |
| **val AUC** | **0.8676** |
| 学習時間 | 2.6 sec ★ 超高速 ★ |

進行:
- iter 100: 0.8597
- iter 200: 0.8649
- iter 300: 0.8668
- iter 400: 0.8677
- iter 443: 0.8676

### LGB+XGB ensemble

```
ens_pred = 0.55 * lgb + 0.45 * xgb
```

| metric | 値 |
|--------|----|
| **ens AUC** | **0.8678** |

### FT-Transformer (GPU、 短縮 epochs=10)

| metric | 値 |
|--------|----|
| device | cuda (RTX 4070 Ti SUPER) |
| epochs | 10 (短縮、 full は 50-100 epochs) |
| batch_size | 512 |
| d_model / heads / layers | 64 / 4 / 3 |
| **val AUC** | **0.8579** |
| 学習時間 | 426.6 sec (~7 min) |
| memory peak | ~15.7GB (16GB の 98%) |

進行 (val AUC):
- ep 1: 0.8196
- ep 3: 0.8496
- ep 5: 0.8546
- ep 7: 0.8522
- ep 10: 0.8579 (上昇継続中、 50-100 epochs で更に改善見込み)

★ FT 10 epochs は under-trained。 v13.5b の full 学習は典型的に 30-50 epochs。 50-100 epochs でも RTX 4070 Ti SUPER で ~30-60 min 視野 ★

### 3-model ensemble (LGB+XGB+FT)

```
ens_pred = 0.50 * lgb + 0.40 * xgb + 0.10 * ft
```

| metric | 値 |
|--------|----|
| **3-model ens AUC** | **0.8676** |

→ FT 10 epochs では 2-model ensemble 0.8678 と差なし。 FT の under-training 影響。

## 解釈

| obs | 意味 |
|-----|------|
| LGB 0.8662 / XGB 0.8676 | 両者ほぼ同等、 単独で V15 base 同等 (cv 違い影響あり) |
| ensemble +0.0002 | LGB と XGB 相関高い、 期待通り |
| 学習時間 < 30 sec | GPU 効果絶大、 full WF も 5-10 min で可能 |

## V15 baseline との比較

| model | AUC | 備考 |
|-------|-----|------|
| V15 (LGB+XGB) | ~0.870 | 全年 WF 平均 |
| V15 (LGB+XGB+FT+IR Grid) | 0.8939 | 4-model 完全版 (CLAUDE.md baseline) |
| V20 quick (LGB+XGB) | 0.8678 | 2022-24 → 2025 単 fold |
| V20 quick (+ FT) | (追記) | 同上 |

→ 単 fold の V20 quick AUC は V15 全年 WF と直接比較不可。 fold 構成が異なる。

## 重要な observation

★ Phase 11/12/13 features (57) を含めても AUC 改善 0、 expected ★。 全 row 同値 → LGB/XGB 自動 drop。

V20 真の改善には:
1. **Phase 11 JRDB 実 data** (5/12+) — gaika / odds_change / jockey ext / paddock features
2. **Phase 12 JV-Link 実 data** (5/24+) — odds 拡張 / 番組 / ハロン / 天候 / 血統
3. **Phase 13 netkeiba master 実 scrape** (5/11+) — AI 展開 / 波乱度 / 個別ラップ / バイアス
4. **4-model ensemble** (FT-Transformer + IntraRace Attention 完全版)

これらが揃うと V20 ensemble AUC 0.91-0.93 視野 (Phase 14 plan 通り)。
