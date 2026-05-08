# Session #55 C: V20 + expanding LGB 学習

**作成**: 2026-05-09 (Session #55 C、 dev/v20-expanding)
**file**: tools/train_v20_expanding.py
**model**: data/v20/models/v20_expanding_v1.pkl
**raw**: data/v18/session_55_v20_expanding_training_raw.json

---

## 1. setup

- 学習データ: jra_races_full.csv 2018-2025 (379,031 rows、 異常除外後)
- target: top3 (1-3着 binary)
- CV: 年度別 3-fold (2023, 2024, 2025 を valid、 train は それ以前 全期間)
- LGB single (FT/IR は Session #56)
- early stopping: 50 round、 max 1000

---

## 2. ★ AUC 結果 (本 PoC の中核) ★

| split | base (16 features) | base + expanding 6 (22 features) | delta |
|-------|-------------------:|---------------------------------:|------:|
| valid 2023 | 0.8084 | 0.8081 | -0.0003 |
| valid 2024 | 0.8129 | 0.8129 | +0.0000 |
| valid 2025 | 0.8111 | 0.8114 | +0.0003 |
| **avg** | **0.8108** | **0.8108** | **-0.0000** |

→ **expanding 6 features 追加で AUC 変化なし (改善 ±0.0003 範囲、 統計的に有意でない)**

---

## 3. feature importance (gain、 full model 最終 fold)

| rank | gain | feature |
|-----:|-----:|---------|
| 1 | 572,086 | popularity |
| 2 | 14,269 | prev_finish |
| 3 | 11,667 | num_horses |
| 4 | 4,814 | rest_days |
| 5 | 4,651 | jockey_wr_lifetime |
| 6 | 4,416 | horse_career_wr_lifetime |
| 7 | 3,922 | age |
| 8 | 3,669 | trainer_top3_lifetime |
| 9 | 3,282 | **trainer_top3_w90 ★** |
| 10 | 2,913 | **jockey_top3_w30 ★** |
| 11 | 2,364 | horse_num_ratio |
| 12 | 2,080 | distance |
| 13 | 1,733 | **horse_career_top3_w10 ★** |
| 14 | 1,532 | **horse_career_wr_w5 ★** |
| 15 | 1,500 | **horse_career_top3_w5 ★** |
| 16 | 1,367 | weight_carry |
| 17 | 1,225 | umaban |
| 18 | 1,194 | **jockey_wr_w30 ★** |
| 19 | 743 | sex_enc |
| 20 | 423 | surface_enc |

**expanding 6 features は全て上位 18 件以内 だが、 既存 lifetime 系と redundant。**

---

## 4. 解釈 (V20 戦略への impact)

### 重要 finding ✅
1. **popularity (確定オッズ由来) が圧倒的 (gain 572K vs 2位 14K = 40倍以上)**
   → pattern A モデルでは popularity は LEAK 扱いだが、 baseline のスコアには大きく貢献している
   → V20 構築では popularity 除外時の真の AUC が baseline (V15 0.8939 = pattern A は popularity 除外済)

2. **expanding features は importance では現役、 だが lifetime 版 と情報重複**
   → jockey_wr_lifetime と jockey_top3_w30 の rank 隣接 (5位 vs 10位)
   → LGB が両者を選んでいるが、 marginal contribution は小さい

3. **V18/V19 sib_w5 +0.0689 は特殊例**
   → sib_top3_rate は元々 lifetime data (post-race)、 リーク含み corr 0.29 だった
   → expanding 化で リーク除去 + 信号維持 (corr 0.20)
   → **対して、 jockey/horse career 系は既に lifetime expanding (cumsum-current) で実装済み = リーク無し**
   → window 限定にしても 信号は変わらず、 微減 (古いデータ捨てる分)

### 結論
- **「単純な expanding 化」 は V20 で AUC 改善なし**
- **本命戦略 を 修正**:
  - ❌ expanding (本 PoC で否定)
  - ❌ 単一 features 追加 (Session #50/51/54 で否定)
  - ✅ **ensemble 強化 (Session #56 FT-Transformer 復活)**
  - ✅ **interaction features (Session #57)**
  - ✅ **target engineering (3着内→着差予測 等)**
  - ✅ **データ拡張 (TFJV 90年分 = Session #44+)**

---

## 5. 5/9 V15 投資保護

✅ V15 model md5 不変
✅ main 不変、 dev/v20-expanding 専用
✅ predict_core / daily_predict / app.py 一切変更なし

---

## 6. 結論

- **expanding 6 features: AUC delta -0.0000 (改善なし)**
- **V18/V19 sib_w5 は特殊例 と確定**
- **V20 構築の 本命戦略 を修正**: expanding 単独 → ensemble + interaction

→ **Session #56 (FT-Transformer 復活) と Session #57 (interaction) が本命**

---

**Session #55 C 完了 (dev/v20-expanding)**
