# Session #54 B: NAR V5 features 拡張候補

**作成**: 2026-05-09 (Session #54 B)
**前提**: A audit (V4 22 features、 AUC 0.8145、 nar_all_races.csv 54K rows)
**目的**: V5 候補 features 確定 + 期待 AUC 整理

---

## 0. 整理方針

V4 = 22 features (Pattern B、 live)。 NAR は paper trade 専用なので Pattern B 維持。
V5 では:
1. **expanding 統計 features** (V4 leak-free 版から移植、 7 件)
2. **NAR data 既取得 features** (V4 未活用、 4 件)
3. **計算 features** (NAR 独自、 4 件)

---

## 1. V5 features 候補 (15 新規)

### 1-1. expanding 統計 (7 件、 V4 leak-free 版から移植)

| # | feature | 計算 | alpha | 期待 AUC |
|---|---------|------|-------|---------|
| 1 | horse_dist_top3r | (horse_id × dist_cat).top3 cumsum / runs | 5 | +0.001-0.002 |
| 2 | horse_surface_top3r | (horse_id × surface_enc).top3 cumsum / runs | 5 | +0.001-0.002 |
| 3 | jockey_course_wr | (jockey × course).win cumsum / runs | 10 | +0.001-0.002 |
| 4 | frame_course_dist_wr | (course × dist × bracket).win cumsum / runs | 50 | +0.0005-0.001 |
| 5 | horse_career_races | horse_id cumcount | -- | +0.0005 |
| 6 | horse_career_wr | horse_id.win cumsum / runs | 5 | +0.001-0.002 |
| 7 | horse_career_top3r | horse_id.top3 cumsum / runs | 5 | +0.001-0.002 |

**小計**: +0.005-0.012

### 1-2. nar_all_races.csv 既取得 (V4 未活用、 4 件)

| # | feature | source col | 内容 | 期待 AUC |
|---|---------|-----------|------|---------|
| 8 | horse_weight_change | horse_weight_change | 当日 馬体重変化 (前走比) | +0.001-0.003 |
| 9 | horse_weight_change_abs | abs(horse_weight_change) | 急変検知 | +0.0005 |
| 10 | last3f_filled | last3f (前走集計) | 前走 上がり 3F (NaN は mean fill) | +0.001-0.002 |
| 11 | trainer_wr | jockey_wr 同様 expanding | 調教師 勝率 | +0.001 |

**小計**: +0.003-0.007

### 1-3. NAR 独自 計算 (4 件)

| # | feature | 計算 | 内容 | 期待 AUC |
|---|---------|------|------|---------|
| 12 | course_dist_wr | (course × dist_cat).win expanding | NAR 場 別 勝率 | +0.001-0.003 |
| 13 | weight_cat_dist | weight_cat × 10 + dist_cat | 体重 × 距離 cross | +0.0005 |
| 14 | nar_class_enc | class_info から encode | NAR 独自 class (C1/C2/B3 等) | +0.0005-0.001 |
| 15 | rest_days_filled | race_date diff (horse_id 単位) | NAR 短間隔 (中央 < ) | +0.0005-0.001 |

**小計**: +0.0025-0.006

---

## 2. V5 全体 features list

```python
NAR_V5_FEATURES = NAR_V4_FEATURES + [
    # expanding
    'horse_dist_top3r', 'horse_surface_top3r',
    'jockey_course_wr', 'frame_course_dist_wr',
    'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
    # nar_all_races 既取得
    'horse_weight_change', 'horse_weight_change_abs',
    'last3f_filled', 'trainer_wr',
    # NAR 独自
    'course_dist_wr', 'weight_cat_dist',
    'nar_class_enc', 'rest_days_filled',
]
# V4 22 + V5 15 = 37 features
```

---

## 3. 期待 AUC

| 段階 | features | 期待 AUC |
|------|---------|---------|
| V4 (現行) | 22 | 0.8145 |
| V5 = V4 + expanding 7 | 29 | 0.820-0.825 (+0.005-0.010) |
| V5 = V4 + expanding + 既取得 4 | 33 | 0.822-0.829 (+0.008-0.015) |
| V5 = V4 + expanding + 既取得 + 独自 4 | 37 | **0.825-0.832** (+0.010-0.018) |

→ V5 目標: **AUC 0.825 以上** (+0.010、 audit 予想 0.82-0.83 と一致)

---

## 4. リーク risk 評価

| feature | risk | 検証 |
|---------|------|----|
| expanding 統計 7 件 | low (cumsum - current で当該レース除外) | dam_top3r 教訓 OK |
| horse_weight_change | low (当日朝発表、 paper trade 範囲) | live OK |
| last3f_filled | medium (前走集計、 race_id sort ↑ 注意) | sort ↑ 必須 |
| trainer_wr | low (expanding) | OK |
| course_dist_wr | low (expanding) | OK |
| nar_class_enc | low (race-level) | pre-race |

→ **全 15 features リーク risk 低 / 中**

---

## 5. 実装 工数

| 領域 | 工数 |
|------|------|
| expanding 7 件 (既存 train_nar_v4_leakfree.py から移植) | 0.5h |
| 既取得 4 件 (read 済 col + dummy pre-process) | 0.5h |
| NAR 独自 4 件 (course_dist_wr + class encoding + rest_days) | 1h |
| LGB + XGB ensemble 学習 (V4 同様) | 0.5h |
| backtest 検証 | 0.5h |
| **合計** | **3h** |

→ Session #54 1 セッション内 完了可能

---

## 6. 5/12 paper trade 投入候補性

✅ NAR は paper trade 専用 → 安全
✅ V5 候補は AUC + 0.005-0.018 期待
✅ V4 並行運用、 A/B test 可能
✅ 5/12 に間に合う (本 Session で V5 model 完成予定)

---

## 7. 結論

✅ V5 候補 features 15 件 確定 (expanding 7 + 既取得 4 + NAR 独自 4)
✅ V5 全体: V4 22 + V5 15 = 37 features
✅ 期待 AUC: 0.825-0.832 (+0.010-0.018)
✅ 5/12 paper trade 投入候補性 高
✅ 工数 3h で 1 セッション内 完了可能

**次 step (Session #54 C)**: tools/train_nar_v5.py 実装 + LGB+XGB 学習 + backtest
