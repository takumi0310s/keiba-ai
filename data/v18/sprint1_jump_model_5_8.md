# Sprint 1 E: 障害レース sub-model PoC (Session #45 E)

**作成**: 2026-05-08 (Session #45 E、 dev/sprint1)
**目的**: 障害レース 専用 LGB model PoC、 平地と独立した予測経路
**ステータス**: ✅ PoC 完了、 BT AUC 0.7536

---

## 1. 構成

### 1.1 data

```
source: data/jra_races_full.csv (532K rows)
filter: race_name に '障害' 含む → 14,257 races
target: is_top3 (3 着以内、 positive rate 30.10%)
```

### 1.2 train / test split

- train: year ≤ 2023 (11,867 races)
- test: year ≥ 2024 (2,390 races)

### 1.3 features (簡易、 10 件)

```python
['weight_carry_num',     # 斤量
 'age_num',              # 年齢
 'num_horses_num',       # 頭数
 'popularity_num',       # ★ 人気 (確定オッズ系、 production では除外検討)
 'distance_num',         # 距離
 'horse_weight_num',     # 馬体重 (Pattern B、 確定後 leak 注意)
 'sex_enc',              # 性別 encoded
 'surface_enc',          # 馬場 encoded (障害は 障芝/障ダ)
 'condition_enc',        # 馬場状態 encoded
 'course_code_num']      # 開催場 (中山/京都/阪神/小倉/新潟/中京)
```

---

## 2. 結果

```
LGB training (early stopping、 30 round patience):
- best iteration: 27
- BT AUC (test 2024+): 0.7536
- training time: 0.2s
- model saved: data/v18/models/v18_jump_lgb.txt
```

→ 平地 V15 (AUC 0.886) より -13pt 低い (障害は予測難しい + features 簡易)

---

## 3. caveat + 制限

### 3.1 popularity 含む = 確定オッズ系 リーク類似

- features に `popularity` (1-18) を含むため、 確定オッズベースの 後追い予測の傾向あり
- production 投入時は popularity を除外して再学習推奨
- 除外時 AUC は 0.65-0.70 程度に下がる見込み

### 3.2 features 簡易

- V15/V19 ほどの features 拡張なし (V15 は 150 features)
- 障害固有 features 未追加:
  - 過去障害成績 (horse 単位)
  - 障害コース適性
  - 騎手の障害成功率
  - 障害飛越上手さ (障害競争 多い騎手?)

→ 5/15 merge 後 + Phase 3 後半 で features 拡張可能

### 3.3 sample 数

- train 11,867 races (障害は 月 30 R 程度)
- 平地 V15 (532K races) に比べ 1/45 の sample
- 障害 sub-model は 「平地 V15 model fallback」 的位置付け

---

## 4. production 統合 plan (5/15 merge 後)

### 4.1 案 1: 障害 R で V15 fallback として使用

```python
# tools/race_auto_notify.py 内
if race_is_jump(race_id):
    # 障害 R → 障害 sub-model で 予測
    p = jump_model.predict(features)
else:
    # 平地 → V15 (現状)
    p = v15_model.predict(features)
```

### 4.2 案 2: V15 + 障害 sub-model アンサンブル

障害 R で V15 + jump model 平均 → 平地 model の障害寄与 強化

### 4.3 ROI 期待

- 障害 R は 過去 退避 (現状 案B改 で除外)
- 投入時 max loss: 700円/R × 障害 R 数 (週末 2-4 R)
- 期待 hit_rate ~30% で ROI 90-110% 想定 (平地 V15 案B改 84% に近い)

---

## 5. 5/9 V15 投資保護

✅ 5/9 朝 障害 R は除外 (現状 案B改 維持)
✅ V15 model md5 不変、 main 不変、 dev/sprint1 only
✅ jump_model.txt は data/v18/models/ 別 dir、 V15 model file に影響なし

→ **5/9 朝 V15 完全保証**

---

## 6. 結論

✅ E1: 14,257 障害 races (1986-2025)、 14 年分 学習 data
✅ E2: BT AUC 0.7536 (test 2024+、 popularity 含む簡易 features)
✅ E3: model file v18_jump_lgb.txt 保存
✅ E4: production 統合は 5/15 merge 後、 popularity 除外 + 障害固有 features 追加で AUC 向上見込み
✅ V15 投資保護

→ **Sprint 1 E PoC 完了、 5/15 merge 後 features 拡張で 本実装**

---

**Session #45 E 完了 (dev/sprint1)**
