# Session #57 C: V20 + interaction LGB 学習 結果

**作成**: 2026-05-09 (Session #57 C)
**実装**: tools/train_v20_interaction.py
**fold**: train 2015-2024、 test 2025 single fold
**target**: is_top3 (複勝圏)
**モデル**: LGB single

---

## 1. 結果サマリー

| 構成 | features | AUC (BT 2025) | logloss | best_iter |
|------|---------|---------------|---------|-----------|
| BASELINE (V15 のみ) | 150 | **0.8685** | 0.3580 | 500 |
| INTERACTION (V15 + 10 int) | 160 | **0.8683** | 0.3581 | 490 |
| **Δ AUC** | +10 | **-0.0002 (-2 bp)** | +0.0001 | -10 |

→ **interaction 10 features は AUC contribution ≈ 0 (-2bp)** - 実質的に redundant

---

## 2. interaction features rank (160 features 中)

| rank | feature | gain | split |
|------|---------|------|-------|
| 42 | int_sire_baba_top3r | 3,828 | 306 |
| 43 | int_trainer_course_top3r | 3,773 | 313 |
| 46 | int_sire_course_top3r | 3,651 | 290 |
| 47 | int_jockey_class_top3r | 3,632 | 290 |
| 48 | int_jockey_trainer_top3r | 3,585 | 292 |
| 60 | int_jockey_distcat_top3r | 3,181 | 251 |
| 66 | int_jockey_baba_top3r | 2,986 | 229 |
| 67 | int_sire_distcat_top3r | 2,984 | 244 |
| 70 | int_jockey_course_top3r | 2,815 | 235 |
| 127 | int_horse_jockey_top3r | 250 | 22 |

### 2.1 観察

- 9/10 が rank 42-70 (中位): 中程度 gain あり、 でも上位を覆さず
- `int_horse_jockey_top3r` は rank 127 (最下層): V15 既存の `jockey_horse_wr` / `jockey_horse_top3r` と完全 redundant
- top 30 に interaction features 1 つも入らず

### 2.2 top 30 features (BASELINE と同じ顔ぶれ)

```
1. paci_jockey_exp_wr     (gain 318k)
2. paci_ninki_idx         (gain 264k)
3. paci_jockey_exp_3rd    (gain 234k)
4. jrdb_ze_idm_avg        (gain 151k)
5. training_time_filled   (gain 88k)
...
```

→ V15 既存の paci / JRDB / 調教 features が dominant、 interaction の入る余地なし

---

## 3. 結論

### 3.1 V15 145 features 飽和の確証

- 単一 feature 追加 (Session #51) で飽和 確認
- ★ **interaction (組み合わせ) でも +0bp** (本 Session) → 真の飽和
- LGB 既に既存 features の組み合わせを内部で捕捉済み (tree boosting の本質)

### 3.2 interaction features は redundant

- V15 既存の expanding rate features (jockey_wr_calc, jockey_course_wr_calc, sire_*_wr 等) が本質的に同じ信号を持つ
- LGB の split で「jockey_id × course_enc → 内部 partition」が暗黙的に実現済み
- → 明示的な interaction feature は冗長 (redundant)

### 3.3 V20 への含意

| アプローチ | 結果 | 次 step |
|-----------|------|---------|
| 単一 feature 追加 | 飽和 (Session #51) | × |
| **interaction 追加** | **飽和 (本 Session)** | × |
| TFJV 新 source 追加 | Session #44 PoC AUC 0.8752 (V19 sib_w5 と同等) | 待 evaluation |
| ensemble (LGB+XGB+FT+IR) | V15 BT 2025 0.8856 = 0.8685 + 0.017 | △ V13.5b 復活 (Session #56) |
| 動画 features | Phase 4 (7-8月) | ◎ 真の breakthrough 候補 |

**判定**: V20 で interaction 追加は不採用。 ensemble または動画 features に注力。

---

## 4. NEXT (Area D)

→ 3-way interaction (jockey × course × dist_cat 等) と shrinkage tuning で再挑戦

---

**Session #57 C 完了**
