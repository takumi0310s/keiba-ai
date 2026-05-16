# AUDIT-1 Executive Summary: 3 source 全要素 audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 J 領域、 統合 doc)
**作業**: read-only audit、 既存 code 完全不変
**目的**: 3 source (JRA-VAN / netkeiba マスター / JRDB) の **全要素 抜けなく audit**、 取得・活用状況 全網羅、 未活用 high-value features 発掘

---

## 0. AUDIT-1 全 doc 一覧

| 領域 | 内容 | doc path |
|------|------|---------|
| A | JRA-VAN / TFJV (14 datatypes / 約 700 fields) | data/v18/audit_jravan_5_8.md |
| B | netkeiba マスターコース (30+ page type) | data/v18/audit_netkeiba_5_8.md |
| C | JRDB (26 datatypes / 約 500 fields) | data/v18/audit_jrdb_5_8.md |
| D | V15 features → source mapping (主要 50 件) | data/v18/audit_v15_features_source_mapping_5_8.md |
| E | 未活用 features Top 30 ROI ranking | data/v18/audit_unused_features_top30_5_8.md |
| E2 | 未活用 features 全リスト (約 130 件) | data/v18/audit_unused_features_full_5_8.md |
| F | 動画 / 画像 features audit | data/v18/audit_video_image_features_5_8.md |
| G | 取得 timing (Tier 0-4) 別 features | data/v18/audit_timing_features_5_8.md |
| H | NAR (地方) audit | data/v18/audit_nar_5_8.md |
| I | 推奨 implementation roadmap | data/v18/audit_roadmap_5_8.md |
| J | 統合 + summary (本 doc) | docs/AUDIT_FULL_REPORT_5_8.md |

---

## 1. 3 source 全要素 一覧

### 1.1 JRA-VAN / TFJV

| カテゴリ | datatype | size | V15 利用率 |
|---------|---------|------|---------|
| レース基本 | RA / SE | 1.9 GB / 4,671 file | ✅ 利用 (jra_races_full.csv 経由) |
| 払戻 | HR | 含 SE | ⚠️ jra_payouts.csv 4/6 停止 |
| オッズ | H1 / H6 | 2.0 GB / 6,160 file | ⚠️ 部分 (H1) / 完全未取得 (H6) |
| 馬個体 | UM | 497 MB / 280 file | ⚠️ 部分 (blood_full 81K) |
| 調教 | CK / TM | 657 MB / 18,089 file | ✅ 利用 (training_times 955K) |
| 繁殖牝馬 | BR | 5.8 MB / 10 file | ❌ 未取得 |
| 生産者 | HS (BS) | 11 MB / 311 file | ❌ 未取得 |
| 馬主 | BN (OW) | 4.1 MB / 10 file | ❌ 未取得 |
| WIN5 | WF | 7.0 MB / 863 file | ❌ 未取得 |
| 出走取消 | JG | 1.7 MB / 41 file | ❌ 未取得 |
| 騎手 master | KS | 単一 | ❌ 未取得 (JRDB KZ で代替) |
| その他 (RC/YS/TK/HN/KT) | -- | -- | ❌ 未取得 |

**14 datatype / 約 700 fields**、 V15 利用率 **約 30%**

### 1.2 netkeiba マスターコース

| カテゴリ | source | V15 組込 |
|---------|--------|--------|
| 基本 (出馬表 / 結果 / db) | scrape_super_premium / horse / jockey 各 db | ✅ 主軸 |
| Premium 系 | speed / oikiri / comment / siblings | ✅ 部分組込 |
| **★ マスター 限定** (master_index / ai_opinion / track_bias / race_analysis 等 11 件) | -- | ❌ ほぼ 全て 未組込 |
| 動画 (重賞 調教 / パドック) | -- | ❌ 未取得 |
| 静止画 (パドック) | -- | ❌ 未取得 |

**30+ page type**、 V15 利用率 **約 30%**、 ★ マスター 限定 11 件 ほぼ 完全未活用

### 1.3 JRDB

| 取得済 (10+ datatypes) | 行数 | V15 組込 |
|---------------------|------|--------|
| KYI | 290K | ✅ 22 件 (主軸) |
| SED | 547K | ✅ 8 件 (前走) |
| TYB | 548K | ✅ 5 件 (Pattern B) |
| PACI / KYI | 548K | ✅ 11 件 (Tier A 7 + Tier B 4) |
| CYB | 548K | ⚠️ 部分 |
| SKB | 547K | ❌ POST-RACE LEAK 確定で完全除外 |
| **SRB** ★最優先 | 21K | ❌ 完全未組込 (★) |
| **JO** | 301K | ❌ 完全未組込 (★) |
| **KKA** | 取得済 | ❌ 完全未組込 (★) |
| **CHA** | 取得済 | ❌ 完全未組込 (★) |
| その他 (UKC/KZ/CZ/KSA/CSA/KAB/KAA/JOA/KTA/BAC/HJC/OZ/OW/OU/OT/OV) | 各取得済 | ⚠️ 部分 / ❌ |

**26 datatype / 約 500 fields**、 V15 利用率 **約 25%**

---

## 2. V15 利用率 (取得済の何 % が利用されているか)

### 2.1 source 別 利用率

| source | 取得済 fields | V15 組込 fields | 利用率 |
|--------|------------|---------------|------|
| JRA-VAN / TFJV | ~700 | ~80 (jra_races_full / training_times 経由) | **11%** |
| netkeiba (Premium + マスター) | 推定 200+ (29 csv) | ~20 | **10%** |
| JRDB | ~500 | ~45 (KYI 22 + SED 8 + PACI 11 + Tier B 4) | **9%** |
| **合計** | **~1,400 fields** | **~145 features** | **約 10%** |

### 2.2 V15 = 150 features の 内訳

- TFJV 主軸 (基本 / ラグ / 集計 / 派生): 約 60 件
- JRDB KYI (前日): 22 件
- JRDB SED (前走): 8 件
- JRDB PACI (Tier A + B): 11 件
- netkeiba (speed_index + training_times + race_lap + siblings_exp): 約 15 件
- v15 新 (jockey_horse + transport + renovation + gaisha): 14 件
- TFJV (Pattern B): 8 件
- 気象庁 / JRA 公式 (Pattern B): 5 件
- (合計 ~145、 派生重複あり = ~150)

---

## 3. 未活用 high-value features Top 10

| # | feature | source | 期待 AUC | 工数 | 投入時期 |
|---|---------|--------|---------|------|---------|
| 1 | **SRB bias_1corner-straight (6 件)** | JRDB SRB | +0.003-0.005 | 4h | Sprint 4 |
| 2 | **netkeiba master_index 5 件** (time/start/chase/agari/master_total) | netkeiba マスター | +0.003-0.005 | 6h | Sprint 4 |
| 3 | **JRDB JO cid_idx / ls_idx 2 件** | JRDB JO | +0.002-0.003 | 3h | Sprint 4 |
| 4 | **JRDB KKA kyori/track/heavy_seiseki** (12 group × 4) | JRDB KKA | +0.002-0.005 | 6h | V20 |
| 5 | **netkeiba ai_opinion pace 予想** | netkeiba マスター | +0.001-0.002 | 2h | Sprint 4 |
| 6 | **JRDB CHA oikiri_rank / idx / 3 time** | JRDB CHA | +0.002-0.005 | 4h | V20 |
| 7 | **netkeiba speed_index dist / course 2 件** | netkeiba | +0.001-0.003 | 1h | Sprint 4 |
| 8 | **JRDB JO gaisha_bb / breeder_bb 6 件** | JRDB JO | +0.001-0.003 | 3h | V20 |
| 9 | **TFJV BS_DATA breeder_top3r** | TFJV BS | +0.002-0.004 | 8h | V20 |
| 10 | **JRDB CYB train_mark / eval / amount 5 件** | JRDB CYB | +0.001-0.003 | 3h | V20 |

**Top 10 期待 AUC 合計: +0.018-0.038** (重複考慮で 実効 +0.010-0.020)

---

## 4. 推奨 next action

### 4.1 即時 (5/9 V15 案B改 維持 後)

5/9 V15 案B改 単独継続 (絶対) → 5/12 から Sprint 4 着手

### 4.2 Sprint 4 (5/12-5/19) 即実装 候補

★★★ 3 件:
1. SRB bias 6 features (4h)
2. master_index 5 features (6h)
3. JO cid_idx / ls_idx 2 features (3h)

合計 13h、 期待 +0.008-0.013 → V15.5 候補 (0.886 → 0.894-0.899)

### 4.3 V20 統合 (5/22-6/8)

★★ 14 features (50h):
- KKA / CHA / 残 JO / CYB / TFJV BS/BN/BR / 残 KYI / 残 TYB / 残 SED / netkeiba ai_position / race_analysis / UKC keito_code / paddock score
- 期待 +0.013-0.030 → V20 (0.890-0.900)

### 4.4 Phase 4 (7/1-8/31) 動画 PoC

5-7 features (80h+):
- パドック静止画 体格 score (1)
- 歩様 stride length / freq / symmetry (3)
- 姿勢 score / head_bobbing (2)
- 緊張度 (1)
- 期待 +0.005-0.010 → V21 (0.900-0.910)

---

## 5. 期待効果 試算

### 5.1 model AUC 改善 path

| 段階 | model | features | WF AUC | 改善 |
|------|------|---------|-------|------|
| 現状 | V15 | 150 | 0.886 (BT 0.894) | -- |
| 5/19 候補 | V15.5 | 156-160 | 0.892-0.896 | +0.006-0.010 |
| 6/8 投入 | V20 | 165-180 | 0.890-0.900 | +0.004-0.014 |
| 6/30 候補 | V20.5 | 180-200 | 0.895-0.910 | +0.005-0.014 |
| 9/1 投入 | V21 (動画) | 185-205 | 0.900-0.915 | +0.005-0.015 |

### 5.2 月利期待 (戦略⑦込み)

| model | ROI | 月利推定 |
|------|-----|---------|
| V15 (現状) | 119-140% | +2-3 万円 |
| V15.5 (5/19) | 130-145% | +3-4 万円 |
| V20 (6/8) | 145-150% | +5-10 万円 |
| V20.5 (6/30) | 150%+ | +6-11 万円 |
| V21 (9/1) | 145-155% | +7-13 万円 |

月額コスト 約 1 万円 (Premium + JRDB + JV-Link + JRA-VAN ネクスト + Colab Pro) は V20 以降で 十分 回収

---

## 6. 5/9 V15 投資保護 (絶対遵守)

✅ 既存 code (predict_core.py / daily_predict.py / app.py / tools/* / train/*) **完全不変**
✅ V15 model file (keiba_model_v15_central*.pkl.gz) **完全不変**
✅ schtasks (DailyPredict / DailyPremiumScrape / DailyResults / WeeklyReport / NightlySanity 等) **完全不変**
✅ dev branch **触らない** (audit doc は main 直 push)
✅ 5/9 朝 V15 daily_predict 完全同一動作 保証
✅ 累計 +5,240 円 維持 (撤退余裕 +55,240 円、 -50,000円 撤退ライン) ※ 旧 +13,530 / +63,530 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)

---

## 7. 結論

### 7.1 audit 完了確認

✅ JRA-VAN / TFJV (14 datatypes / 約 700 fields) 完全 audit
✅ netkeiba マスターコース (30+ page type) 完全 audit
✅ JRDB (26 datatypes / 約 500 fields) 完全 audit
✅ V15 features → source mapping (主要 50 件) 完成
✅ 未活用 features 全リスト (約 130 件) + Top 30 ROI ranking 完成
✅ 動画 / 画像 + 取得 timing + NAR + roadmap 各 audit 完成

### 7.2 主要 発見

1. **V15 利用率 約 10%** (取得済 1,400 fields の うち 150 features 採用)
2. **★ JRDB SRB / 4 datatypes が完全 未組込**: SRB / JO / KKA / CHA 期待 +0.010-0.020
3. **★ netkeiba マスター限定 11 件 ほぼ 完全未活用**: master_index / ai_opinion / ai_position / track_bias 等 期待 +0.008-0.015
4. **★ TFJV BS / BN / BR (生産者・馬主・繁殖牝馬) 完全未取得**: breeder_top3r / owner_top3r で 期待 +0.005-0.010
5. **POST-RACE LEAK 教訓**: SKB 全 10 features 完全除外、 dam_top3r 静的版 / 印 系 7 件 リーク check 必須

### 7.3 Sprint 4 推奨

★★★ 3 件 (13h) で **V15.5 (0.894-0.899) 候補**:
1. SRB bias 6 件
2. master_index 5 件
3. JO cid_idx / ls_idx 2 件

検証 step: WF 6-fold + リーク check + LIVE retro 1 開催

---

## 8. AUDIT-1 完了

✅ A-J 各領域 doc 完成 (10 + 1 doc)
✅ 3 source 全要素 audit
✅ 未活用 features Top 30 + 全 130 件 list 完成
✅ Sprint 4 / V20 / V21 / Phase 4 への path 完成
✅ 既存 code / model / schtasks 完全不変、 V15 投資保護 ✅

**AUDIT-1 完了、 3 source 全要素 audit、 未活用 features Top 30 発見**
