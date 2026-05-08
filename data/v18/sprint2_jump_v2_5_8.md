# Sprint 2 F: jump_race_model v2 (Session #47 F)

**作成**: 2026-05-08 (Session #47 F、 dev/sprint2)
**前提**: Sprint 1 E AUC 0.7536 (popularity 含む = リーク類似)

---

## 1. v2 改善点

### 1.1 popularity 除外 ★

Sprint 1 E は features に `popularity` (人気) を含めていたが、 これは **確定オッズ系 リーク類似** (締切時オッズから計算)。

→ v2 では完全除外、 production candidate に格上げ

### 1.2 障害固有 features 3 件追加

```python
- horse_jump_top3_rate_exp  # 馬の障害成功率 expanding (alpha=5、 prior 0.30)
- is_grand_jump             # 大障害 (中山GJ 等) 判定
- jockey_jump_top3_rate_exp # 騎手の障害成功率 expanding
```

---

## 2. 結果

| version | features | popularity | AUC | 用途 |
|---------|---------|-----------|-----|------|
| Sprint 1 E (旧) | 10 | 含む (リーク) | 0.7536 | 廃止 (リーク) |
| **Sprint 2 F (v2)** | **13** | **除外** | **0.6778** | **★ production candidate** |

→ popularity 除外で AUC -0.0758 (期待 0.65-0.70 範囲内)
→ リーク 除去後の **真の AUC**

---

## 3. 障害固有 features 寄与確認

LGB feature importance 推定 (model 内部):
- horse_jump_top3_rate_exp: 高寄与 (期待)
- jockey_jump_top3_rate_exp: 中寄与
- is_grand_jump: 低寄与 (data 少)

---

## 4. production 統合 plan (5/15 merge 後)

```python
# tools/race_auto_notify.py
if race_is_jump(race_id):
    p = jump_v2_model.predict(features)  # AUC 0.6778
else:
    p = v15_model.predict(features)  # 平地 AUC 0.886
```

V15 案B改 は平地 1勝のみのため、 障害は別経路:
- 5/22+ Phase 3 で 障害投入候補検討
- 期待 hit_rate 30% × ROI 90-110% 想定

---

## 5. caveat + 制限

- AUC 0.6778 は 平地 V15 (0.886) より大幅低、 障害は単独 model 限界
- features 拡張余地:
  - 落馬 history (TFJV から取得)
  - 障害 種別 (大障害 / 平地障害) 詳細
  - 騎手の落馬率

→ Phase 3 後半で 強化 plan

---

## 6. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only

→ **5/9 朝 V15 完全保証**、 5/9 障害 R は除外維持

---

**Session #47 F 完了 (dev/sprint2)**
