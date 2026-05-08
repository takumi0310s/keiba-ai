# Sprint 2 D: maiden_race_model PoC (Session #47 D)

**作成**: 2026-05-08 (Session #47 D、 dev/sprint2)

---

## 1. 構成

source: `data/jra_races_full.csv` class_code=15 (新馬戦) = **43,959 races**

```
target: is_top3 (positive rate 22.99%)
train: year ≤ 2022 (32,014 races)
test:  year ≥ 2023 (11,945 races)
features: 15 (numerical 7 + categorical 4 + 血統 3 + course)
```

血統 3 features:
- father_enc (父)
- mother_enc (母)
- bms_enc (母父)

---

## 2. 結果

```
LGB BT AUC (test 2023+): 0.8092
training time: 0.9 sec
model saved: data/v18/models/v18_maiden_lgb.txt
```

→ V19 平地 0.8754 より -7pt 低い (新馬戦は data 少、 過去成績 nil)
→ 血統 + popularity ベースの 単純 model としては 妥当 AUC

---

## 3. caveat + 制限

- popularity 含む = 確定オッズ系 リーク類似
- 新馬戦 features 拡張余地:
  - 父系 expanding ROI (TFJV UM 90 年分)
  - 母系 全兄弟成績
  - 育成厩舎 (trainer 経験)
  - 騎乗 騎手 新馬戦 record
- production では Phase 3 後半 (5/16-6/8) で 強化

---

## 4. production 統合 plan (5/15 merge 後)

```python
if class_code == 15:  # 新馬戦
    p = maiden_model.predict(features)
else:
    p = v15_model.predict(features)  # 平地
```

---

## 5. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**、 5/9 新馬戦は現状除外維持

---

**Session #47 D 完了 (dev/sprint2)**
