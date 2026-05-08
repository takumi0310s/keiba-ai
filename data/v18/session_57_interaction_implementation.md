# Session #57 B: V20 interaction features 実装結果

**作成**: 2026-05-09 (Session #57 B)
**実装**: tools/v20_interaction_features.py
**output**: data/v20/interaction_features.csv (527,280 行, 65 MB)

---

## 1. 実装概要

### 1.1 計算 logic

```
1. df = pickle.load(data/_v15_train_df_cache.pkl)  # 527,280 行
2. df.sort_values(['date_num','race_id'])  # 時系列 sort
3. for each (name, keys, alpha) in INTERACTION_SPEC:
     cum_sum = df.groupby(keys)['is_top3'].cumsum() - df['is_top3']  # 当該レース除外
     cum_cnt = df.groupby(keys).cumcount()  # 過去レース数 (0-indexed)
     feat = (cum_sum + alpha * prior) / (cum_cnt + alpha)  # Bayesian smoothing
4. → data/v20/interaction_features.csv (race_id, horse_id, 10 features)
```

### 1.2 リーク防止

- ✅ `cumsum() - df['is_top3']`: 当該レース除外 (cumsum-current pattern)
- ✅ `date_num` 昇順 sort: 時系列順保証
- ✅ Bayesian smoothing: 件数 0 → prior (global mean) で初期化

---

## 2. 結果 (10 features 統計)

| feature | alpha | mean | std | nonzero |
|---------|-------|------|------|---------|
| int_horse_jockey_top3r | 3 | 0.2472 | 0.1070 | 527,280 |
| int_jockey_course_top3r | 10 | 0.2136 | 0.0955 | 527,280 |
| int_jockey_distcat_top3r | 10 | 0.2129 | 0.0941 | 527,280 |
| int_jockey_baba_top3r | 5 | 0.2110 | 0.0952 | 527,280 |
| int_jockey_class_top3r | 5 | 0.2113 | 0.0978 | 527,280 |
| int_trainer_course_top3r | 10 | 0.2171 | 0.0713 | 527,280 |
| int_sire_course_top3r | 30 | 0.2193 | 0.0450 | 527,280 |
| int_sire_distcat_top3r | 30 | 0.2205 | 0.0474 | 527,280 |
| int_sire_baba_top3r | 20 | 0.2199 | 0.0441 | 527,280 |
| int_jockey_trainer_top3r | 5 | 0.2282 | 0.1069 | 527,280 |

global prior (is_top3 mean): **0.2164**

### 2.1 解釈

- mean ≈ prior (0.2164) → リークなし、 prior 中心
- std: jockey × horse / jockey × trainer が最大 (0.107) → 識別力高
- std: sire × baba (0.044) → shrinkage 強で安定 (alpha=20)
- 全 527,280 行で nonzero (Bayesian smoothing で件数 0 でも prior 入る)

### 2.2 horse_jockey は v15 既存と差別化

V15 既存の `jockey_horse_wr` (alpha=3) / `jockey_horse_top3r` (alpha=3) と alpha は同じ。
本 PoC では target を **top3** に絞り、 raw key (horse_id × jockey_id) で再計算。
LGB 学習で同等 importance なら新規 feature は冗長 → C で重要度確認。

---

## 3. 性能

| 項目 | 値 |
|------|----|
| 入力 | 527,280 行 × 233 cols (917 MB pkl) |
| 計算時間 | 8.0 秒 (10 features) |
| 出力 | 65 MB CSV |
| メモリ | ~3 GB peak |

→ Phase 3 学習 pipeline に余裕で組み込み可能 (10秒未満)

---

## 4. NEXT (Area C)

→ tools/train_v20_interaction.py で V20 base + 10 features を LGB 学習
→ V20 alone vs V20+interaction の AUC 比較

---

**Session #57 B 完了**
