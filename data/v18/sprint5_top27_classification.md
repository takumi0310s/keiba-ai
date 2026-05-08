# Session #51 A: AUDIT-1 Top 27 (★★★ 除く) 分類

**作成**: 2026-05-08 (Session #51 A)
**前提**: AUDIT-1 E (audit_unused_features_top30_5_8.md) Top 30 の 内 ★★★ 3 件 (#1-3) を除く 27 件 (#4-30)
**位置付け**:
- ★★★ 3 件 (#1 SRB bias / #2 master_index / #3 jrdb_jo cid_idx) → Sprint 4 (Session #50 並行) で実装中
- 本 doc 27 件 → Sprint 5 / 6 / V20 / Phase 4 振り分け

---

## 0. 分類軸

| 分類 | 条件 | 対象 Sprint |
|------|------|----------|
| **即実装可能** | CSV 取得済 + parser 完成 + 工数 ≤ 4h + リーク risk low/medium | Sprint 5 (5/16-5/22) 候補 |
| **中期** | TFJV parser 必要 / UKC 未取得 / expanding 化 必要 / 工数 4-8h | Sprint 6 (5/23-5/30) or V20 統合 (5/22-6/8) |
| **長期** | リアルタイム化 / 画像解析 / 工数 80h+ | Phase 4 (7-9月) |

---

## 1. 即実装可能 (18 件、 Sprint 5 候補)

| # | feature | source | CSV 存在 | 工数 | 期待 AUC | リーク | コメント |
|---|---------|--------|---------|------|---------|------|--------|
| 4 | jrdb_kka kyori/track/heavy_seiseki | jrdb_kka.csv | ✓ | 6h | +0.002-0.005 | low (expanding 必要) | 12 group × 4 着回数 |
| 5 | ai_opinion pace | netkeiba_ai_opinion.csv | ✓ | 2h | +0.001-0.002 | pre | front/even/slow 予想 |
| 6 | jrdb_cha oikiri_rank/idx/3 time | jrdb_cha.csv | ✓ | 4h | +0.002-0.005 | pre | 調教 詳細 |
| 7 | speed_index dist/course | netkeiba_speed_index.csv | ✓ | 1h | +0.001-0.003 | pre | **最速実装** |
| 8 | jrdb_jo gaisha_bb / breeder_bb | jrdb_jo.csv | ✓ | 3h | +0.001-0.003 | pre | 外厩 BB / 生産者 BB |
| 9 | ai_position 位置取り pct | netkeiba_ai_position.csv | ✓ | 2h | +0.001-0.002 | pre | left/top pct |
| 10 | jrdb_cyb train_mark/eval/amount/change | jrdb_cyb.csv | ✓ | 3h | +0.001-0.003 | pre | 調教分析 5 |
| 11 | training_times rank A/B/C/D | netkeiba_training_times.csv | ✓ | 1h | +0.001 | pre | **最速実装** |
| 12 | race_analysis 馬別 score | netkeiba_race_analysis.csv | ✓ | 2h | +0.001-0.002 | pre | comment + score |
| 17 | jrdb_tyb cancel_flag / bagu_change | jrdb_tyb.csv | ✓ | 2h | +0.001-0.002 | live | Pattern B 専用 |
| 18 | jrdb_sed time_sec/first_3f/last_3f | jrdb_sed.csv | ✓ | 3h | +0.001-0.003 | pre | 前走 detailed time (SED 修復後) |
| 19 | jrdb_kyi 印 残 3 | jrdb_kyi.csv | ✓ | 1h | +0.0003-0.001 | pre | 情報/厩舎/激走印 |
| 21 | netkeiba upset_level/top_pop_reliability | netkeiba_upset_level.csv | ✓ | 1h | +0.0003-0.001 | pre | 波乱度 |
| 22 | race_review prev_review_score | netkeiba_race_review.csv | ✓ | 2h | +0.0001-0.001 | pre | v12.1 不採用後 再検討 |
| 23 | stable_comment_score | netkeiba_stable_comments.csv | ✓ (カバレッジ要確認) | 1h | +0.0003-0.001 | pre | カバレッジ 60%+ なら採用 |
| 24 | jrdb_kz/cz year_leading | jrdb_kz.csv | ✓ | 3h | +0.0005-0.001 | pre | 騎手・調教師 leading |
| 26 | jrdb_jo soten_odds / yoso_odds | jrdb_jo.csv | ✓ | 2h | +0.001 | pre (gap 検証必須) | 朝オッズ近似 |
| 28 | netkeiba data_analysis category × value | netkeiba_data_analysis.csv | ✓ | 2h | +0.0005-0.001 | pre | テキスト解析 |

**合計**: 18 件、 工数 38h、 期待 AUC +0.014-0.041

---

## 2. 中期 (7 件、 Sprint 6 / V20 統合)

| # | feature | source | 状態 | 工数 | 期待 AUC | コメント |
|---|---------|--------|-----|------|---------|--------|
| 13 | jrdb_ukc keito_code/owner_code | JRDB UKC | **未取得** | 3h+取得 | +0.001-0.003 | UKC 取得必要 |
| 14 | TFJV BS_DATA breeder_top3r | TFJV BS | parser 流用 (Session #44 B) | 8h | +0.002-0.004 | dam_top3r 教訓、 expanding 必須 |
| 15 | TFJV OW_DATA owner_top3r | TFJV BN | parser 流用 | 8h | +0.002 | expanding 必須 |
| 16 | TFJV BR_DATA dam 拡張 (sib_*_ext) | TFJV BR | parser 流用 | 6h | +0.001-0.003 | 90 年分 母系 |
| 20 | TFJV RA youbi / direction / weight_type | TFJV RA | jrdb_kab/bac 一部代替 | 2h | +0.0003-0.001 | 補完 |
| 25 | TFJV WF (WIN5) appearance_count | TFJV WF | parser 流用 | 4h | +0.0005-0.001 | 10 年分 WIN5 |
| 27 | TFJV TM_DATA | TFJV TM | 用途要確認 | 6h | unknown | 調査必要 |

**合計**: 7 件、 工数 37h、 期待 AUC +0.0066-0.014

---

## 3. 長期 (2 件、 Phase 4 候補)

| # | feature | source | 状態 | 工数 | 期待 AUC | コメント |
|---|---------|--------|-----|------|---------|--------|
| 29 | TFJV JG (出走取消) リアルタイム | TFJV JG | 取消検知 自動化 | 6h | リアルタイム反映 質改善 | live, Pattern B |
| 30 | パドック画像解析 | netkeiba paddock images | YOLOv8 環境 動作確認済 (Session #43 D) | 80h+ | +0.005-0.010 | Phase 4 (7-9月) 候補 |

**合計**: 2 件、 工数 86h+、 期待 AUC +0.005-0.010 (画像 PoC 成功時)

---

## 4. Sprint 5 推奨 即着手 候補 (上位 5)

期待 AUC × 工数効率 で Sprint 5 (5/16-5/22) 着手:

| 順 | # | feature | 工数 | 期待 AUC | 効率 (AUC/h) |
|----|---|---------|------|---------|-------------|
| 1 | 7 | speed_index dist/course | 1h | +0.001-0.003 | 0.001-0.003/h |
| 2 | 11 | training_times rank | 1h | +0.001 | 0.001/h |
| 3 | 4 | jrdb_kka 12group seiseki | 6h | +0.002-0.005 | 0.0003-0.0008/h |
| 4 | 6 | jrdb_cha 調教詳細 | 4h | +0.002-0.005 | 0.0005-0.0013/h |
| 5 | 10 | jrdb_cyb train_mark | 3h | +0.001-0.003 | 0.0003-0.001/h |

**合計** 15h で +0.007-0.017 期待。 Sprint 5 1 週間 で 完了可能。

---

## 5. 全体集計

| 分類 | 件数 | 工数 | 期待 AUC |
|------|----|------|---------|
| 即実装可能 (Sprint 5) | 18 | 38h | +0.014-0.041 |
| 中期 (Sprint 6 / V20) | 7 | 37h | +0.0066-0.014 |
| 長期 (Phase 4) | 2 | 86h+ | +0.005-0.010 |
| **合計** | **27** | **161h+** | **+0.025-0.065** |

V15 baseline 0.8939 + 0.025-0.065 = **0.919-0.959** (理論 上限、 重複あり)

実効では V18/V19 (sib_*_exp) → V20 (TFJV 統合) → V21 (動画) で 段階投入。

---

## 6. 結論

✅ 27 件 を 3 分類:
- 即実装可能 18 件 → Sprint 5 候補 (38h で +0.014-0.041)
- 中期 7 件 → Sprint 6 / V20 統合候補 (37h で +0.0066-0.014)
- 長期 2 件 → Phase 4 候補 (Phase 4 80h+ 別予算)

**次 step**: Session #51 B で 即実装可能 18 件 を 一括 backtest、 実 AUC contribution を 確認.
