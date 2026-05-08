# Session #51 C: AUDIT-1 combo search 結果 (Top 5)

**作成**: 2026-05-08 (Session #51 C)
**tool**: tools/audit_combo_search.py
**JSON**: data/v18/sprint5_combo_metrics.json

---

## 0. 設計

- B 結果から LEAK / parser 不全 / coverage 0% を 除外、 残り 6 features で 全 pair (6C2 = 15) 検証
- V15 baseline AUC 0.86812 (200 rounds、 2024 valid)
- 各 combo: V15 + 2 features を LGB 学習 → AUC delta 計測
- 期待: 単一で delta ~0 でも、 2 features の **相乗効果** で +0.001 期待

候補 6 features (B 結果 delta が 大きい / 関連 領域):
- #8 jrdb_jo_bb (外厩 BB)
- #6 jrdb_cha_oikiri (調教詳細)
- #17 jrdb_tyb_live (直前 live)
- #10 jrdb_cyb_train (調教分析)
- #12 race_analysis_score (馬別 score)
- #7 speed_index_dist_course (距離・コース指数)

---

## 1. Top 5 結果

| 順 | combo | features | AUC base | AUC combo | delta | コメント |
|----|-------|----------|----------|-----------|-------|--------|
| 1 | [#8, #6] | jo_bb + cha_oikiri | 0.86812 | 0.86801 | **-0.00011** | 最も近い但し負 delta |
| 2 | [#17, #7] | tyb_live + speed_idx | 0.86812 | 0.86790 | -0.00021 | live + index、 補完なし |
| 3 | [#10, #12] | cyb_train + race_analysis | 0.86812 | 0.86775 | -0.00037 | 調教 + score |
| 4 | [#17, #10] | tyb_live + cyb_train | 0.86812 | 0.86772 | -0.00040 | 調教系 重複 |
| 5 | [#8, #17] | jo_bb + tyb_live | 0.86812 | 0.86767 | -0.00045 | 外厩 + 直前 |

**重要**: **全 15 combo で delta ≤ 0** (上 5 でも 0 達成不可)

---

## 2. 結論

✅ V15 base 145 features は **既に 高度 飽和**
✅ 2-feature combo では V15 superset は 構築不可
✅ Sprint 5 / V20 統合 アプローチ修正必要

**主結論**:
1. **V15.5 / V15.6 の 単純追加 path は 効果薄**
2. 大規模 features (TFJV BS/BN/BR、 KKA parser fix) で **直交情報** 投入 必要
3. 動画 features (Phase 4) は **異 modal** で 高 効果可能性

---

## 3. V15 飽和の 解釈

V15 145 features が 高度な 統計集約 (expanding window、 jockey × course、 sire × distance、
trainer × class 等) を 含み、 線形/低次 相互作用は 既に 学習済み。 単純追加は
**冗長** に なる。

→ V20 で 必要な 工夫:
- **領域違い** features (TFJV 90 年血統、 owner、 breeder)
- **画像 modal** (パドック、 走路 動画)
- **時系列** features (前走系の deeper temporal)

---

## 4. Sprint 5/6/V20 への 影響

| Sprint | 元方針 | 修正後 |
|--------|------|------|
| Sprint 5 | 18 件 即実装 | text encode 修正 + Pattern B 改善 のみ (小規模) |
| Sprint 6 | 中期 7 件 | KKA parser 修復 + TFJV BS/BN parser 統合 (本命) |
| V20 (5/22-6/8) | TFJV 統合 | breeder_top3r + owner_top3r + dam_top3r_ext (expanding 化必須) |
| Phase 4 (7-9月) | 動画 PoC | パドック画像 (異 modal) → 効果期待大 |
