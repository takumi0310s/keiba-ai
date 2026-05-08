# AUDIT-1 E: 未活用 features Top 30 ROI ranking (5/8)

**作成**: 2026-05-08 (AUDIT-1 E 領域)
**前提**: A (TFJV) + B (netkeiba) + C (JRDB) audit 結果から ROI rank
**位置付け**: Sprint 4 候補 + V20 統合候補 + Phase 4 候補 を 1 表に集約

---

## 0. 評価軸

- **期待 AUC contribution**: 高 (+0.005-0.01) / 中 (+0.001-0.005) / 低 (< +0.001)
- **実装工数**: h 単位
- **dependency**: 他 features 必要な前提
- **リーク risk**: post-race / pre-race / unknown

---

## 1. ROI ranking Top 30

| # | feature | source | datatype / column | 期待 AUC | 工数 | リーク | 採用優先度 | 備考 |
|---|---------|--------|------------------|---------|------|------|----------|------|
| 1 | bias_1corner-straight (6) | JRDB | SRB | +0.003-0.005 | 4h | pre | ★★★ | track_bias の真値、 Sprint 4 最優先 |
| 2 | master_index 5 indices | netkeiba マスター | master_index.csv (time/start/chase/agari/master_total) | +0.003-0.005 | 6h | pre | ★★★ | マスター限定 取得済 / V15 完全未活用 |
| 3 | jrdb_jo cid_idx / ls_idx | JRDB | JO | +0.002-0.003 | 3h | pre | ★★★ | 取得済 / 数値指数 系 |
| 4 | jrdb_kka kyori/track/heavy_seiseki | JRDB | KKA (12 group × 4) | +0.002-0.005 | 6h | pre | ★★ | 距離・トラック・重 別 着回数 |
| 5 | netkeiba ai_opinion pace | netkeiba マスター | ai_opinion.csv | +0.001-0.002 | 2h | pre | ★★ | pace 予想 (front/even/slow) |
| 6 | jrdb_cha oikiri_rank/idx/3 time | JRDB | CHA | +0.002-0.005 | 4h | pre | ★★ | 調教 詳細 |
| 7 | speed_index dist / course 別 | netkeiba | speed_index.csv (index_dist / index_course) | +0.001-0.003 | 1h | pre | ★★ | 取得済 / 即追加可 |
| 8 | jrdb_jo gaisha_bb / breeder_bb (6) | JRDB | JO | +0.001-0.003 | 3h | pre | ★★ | 外厩 BB / 生産者 BB |
| 9 | netkeiba ai_position 位置取り | netkeiba マスター | ai_position.csv | +0.001-0.002 | 2h | pre | ★★ | 位置取り pct |
| 10 | jrdb_cyb train_mark/eval/amount/change (5) | JRDB | CYB | +0.001-0.003 | 3h | pre | ★★ | 調教分析 |
| 11 | training_times rank A/B/C/D | netkeiba | training_times.rank | +0.001 | 1h | pre | ★★ | 取得済 / 5 段 |
| 12 | netkeiba race_analysis 馬別 score | netkeiba マスター | race_analysis.csv | +0.001-0.002 | 2h | pre | ★★ | comment + score |
| 13 | jrdb_ukc keito_code / owner_code | JRDB | UKC | +0.001-0.003 | 3h | pre | ★★ | 系統 + 馬主 |
| 14 | TFJV BS_DATA breeder_top3r | TFJV BS | (parser 経由) | +0.002-0.004 | 8h | pre | ★★ | 生産者 expanding、 V20 候補 |
| 15 | TFJV OW_DATA owner_top3r | TFJV BN | (parser 経由) | +0.002 | 8h | pre | ★★ | 馬主 expanding、 V20 候補 |
| 16 | TFJV BR_DATA dam 拡張 (sib_*_ext) | TFJV BR | (parser 経由) | +0.001-0.003 | 6h | pre | ★★ | 90 年分 母系、 V20 候補 |
| 17 | jrdb_tyb cancel_flag / bagu_change | JRDB | TYB | +0.001-0.002 | 2h | pre (live) | ★★ | 直前情報、 Pattern B |
| 18 | jrdb_sed time_sec / first_3f / last_3f (前走 詳細時計) | JRDB | SED | +0.001-0.003 | 3h | pre | ★ | 前走 detailed time |
| 19 | jrdb_kyi 印 7 件 (採用済 4 残 3) | JRDB | KYI | +0.0003-0.001 | 1h | pre | ★ | 残り 印 (情報 / 厩舎 / 激走印) |
| 20 | TFJV RA youbi / direction / weight_type | TFJV | RA | +0.0003-0.001 | 2h | pre | ★ | (jrdb_kab/bac で代替済 一部) |
| 21 | netkeiba upset_level / top_pop_reliability | netkeiba | upset_level.csv | +0.0003-0.001 | 1h | pre | ★ | 波乱度 |
| 22 | netkeiba race_review prev_review_score | netkeiba | race_review.csv | +0.0001-0.001 | 2h | pre | ★ | v12.1 不採用後 再検討 |
| 23 | netkeiba stable_comment_score | netkeiba | stable_comments.csv | +0.0003-0.001 | 1h | pre | ★ | カバレッジ 60% 改善後 再評価 |
| 24 | jrdb_kz/cz year_leading 系 | JRDB | KZ/CZ | +0.0005-0.001 | 3h | pre | ★ | 騎手・調教師 leading |
| 25 | TFJV WF (WIN5) appearance_count | TFJV WF | (parser 経由) | +0.0005-0.001 | 4h | pre | ★ | 10 年分 WIN5 |
| 26 | jrdb_jo soten_odds / yoso_odds | JRDB | JO | +0.001 | 2h | pre | ★ | 朝オッズ近似 |
| 27 | TFJV TM_DATA (TM 直 利用) | TFJV TM | (parser 経由) | unknown | 6h | pre | ★ | 用途要確認 |
| 28 | netkeiba data_analysis / race_tendency | netkeiba マスター | data_analysis.csv | +0.0005-0.001 | 2h | pre | ★ | category × value |
| 29 | TFJV JG (出走取消) リアルタイム | TFJV JG | (parser 経由) | リアルタイム反映の 質 改善 | 6h | live | ★ | 取消検知 自動化 |
| 30 | パドック画像解析 (体格 / 緊張度) | netkeiba | paddock images | +0.005-0.010 | 80h+ | live | ★ | Phase 4 候補 (画像 PoC) |

---

## 2. 採用優先度 サマリ

| 優先度 | features 件数 | 期待合計 AUC | 累計工数 |
|-------|------------|-------------|---------|
| ★★★ (Sprint 4 即実装) | #1-3 (3 件) | +0.008-0.013 | 13h |
| ★★ (V20 統合候補、 5/16-6/8) | #4-17 (14 件) | +0.013-0.030 | 50h |
| ★ (V21 候補、 6/8 以降) | #18-30 (13 件) | +0.005-0.020 | 40h+ (Phase 4 80h+ 別) |
| **合計** | **30 件** | **+0.026-0.063** | **103-180h** |

---

## 3. リーク risk 評価

| feature | risk | 検証方法 |
|---------|------|--------|
| paci_ninki_idx | medium (V14.1 で 採用済、 確認済) | 既 v14.1 で gap 検証済 |
| jrdb_jo soten_odds | medium (基準オッズ) | 6-fold WF で gap < 0.05 確認 |
| KKA seiseki 系 | low (集計値、 expanding ベース 計算 必要) | dam_top3r 教訓、 expanding 化 必須 |
| 全 印 系 (7 件) | high (POST-RACE risk) | SKB 教訓、 corr_target / monotonic check 必須 |
| breeder_top3r | low (expanding) | dam_top3r と同根、 expanding 化 必須 |
| owner_top3r | low (expanding) | 同上 |
| WIN5 appearance_count | low | 出走 の 事実は pre-race |

---

## 4. dependency

| feature | 前提 |
|---------|------|
| #1 SRB bias | jrdb_srb.csv 取得済 |
| #2 master_index 5 indices | netkeiba_master_index.csv 取得済 |
| #4 KKA seiseki 系 | jrdb_kka.csv 取得済 |
| #14-16 TFJV (BS/BN/BR) | tfjv_parser.py 完成 (Session #44 B) |
| #25 WF | tfjv_parser.py + W5_DATA |
| #29 JG リアルタイム | parser + scrape リアルタイム化 |
| #30 パドック画像 | YOLOv8 環境 (Session #43 D 動作確認済) |

---

## 5. 結論

✅ Top 30 未活用 features ROI ranking 完成
✅ ★★★ 3 件 (期待 +0.008-0.013、 13h): Sprint 4 即実装候補
✅ ★★ 14 件 (期待 +0.013-0.030、 50h): V20 統合候補
✅ ★ 13 件: 中長期 + Phase 4 候補

**Sprint 4 即実装 (今週中) を 採用すれば V15 0.886 → V15.5 0.894 ~ 0.899 期待**
