# Phase 10 C: JRDB Advance 完全 audit (5/10)

> Session #87 (2026-05-10 夜) Phase 10 C 領域
> 対象: ★ JRDB Advance (¥2,880/月、 既加入) ★
> 趣旨: read-only audit、 V15 production 完全不変

---

## 1. 加入サービス概要

| 項目 | 値 |
|------|----|
| サービス名 | JRDB Advance |
| 月額 | ¥2,880 (税込) |
| 加入状況 | ✅ 加入済 (継続中) |
| 取得経路 | tools/jrdb_health_check.py + AM6:00 daily DL |
| 既保存 dir | data/jrdb/extracted/{Bac, Cha, Cs, Cyb, Hjc, Jo, Kaa, Kab, Kka, Ks, Kta, Kyi, Ot, Ou, Ov, Ow, Oz, Paci, Sed, Skb, Tyb, Ukc} |
| 期間 | 2015-2026 (11 年分) |

---

## 2. 全 26 種 (実保存 22 種) 完全 list

### 2.1 入力データ (3 種)

| code | 内容 | V15 既統合 | 用途 |
|------|------|-----------|------|
| BAC | 番組 | ⚠ 部分 | レース program info |
| HJC | 払戻 | ❌ 未統合 | 払戻 (jra_payouts.csv 補完候補) |
| OT/OU/OV/OW/OZ | 基準オッズ | ❌ 未統合 | オッズ各種 |

### 2.2 メイン前日データ (4 種)

| code | 内容 | V15 既統合 | features |
|------|------|-----------|----------|
| ★ KYI ★ | 競走馬 (主軸) | ✅ 完全統合 | jrdb_idm, training_idx, stable_idx, info_idx, composite_idx, upset_idx, ten/pace/agari/position_idx_pred, class_code, rise_code, heavy_apt, hoof_code, ranch_rank, stable_rank, entry_days_ago, entry_race_num, training_arrow, stable_eval, running_style, dist_apt = **22 features** |
| KKA | 情報 (CID 素点) | ⚠ 部分 (jrdb_kka_features.csv 経由) | CID 詳細 |
| CS/CSA | 調教分析 | ❌ 未統合 | 変化指標 |
| CHA | 調教本追切 | ❌ 未統合 | 本追切詳細 |

### 2.3 マスタ (3 種)

| code | 内容 | V15 既統合 |
|------|------|-----------|
| UKC | 馬基本 | ⚠ 部分 (blood_full.csv 経由) |
| KSA / KS | 騎手マスタ | ⚠ 部分 (騎手成績 jrdb_features 経由) |
| KTA / Kta | 調教師マスタ | ⚠ 部分 |
| JOA / Jo | 情報 (?) | ❌ 未統合 |

### 2.4 リアルタイム (1 種)

| code | 内容 | V15 既統合 | features |
|------|------|-----------|----------|
| ★ TYB ★ | 直前データ | ✅ 完全統合 | jrdb_paddock_idx, odds_idx, live_composite_idx, body_code, demeanor_code = **5 features** (Pattern B 専用) |

### 2.5 履歴 (2 種)

| code | 内容 | V15 既統合 | features |
|------|------|-----------|----------|
| ★ SED ★ | 成績 | ✅ 完全統合 | jrdb_prev_idm, prev_track_bias, prev_interference, prev_late_start, prev_ten_idx, prev_agari_idx, prev_pace_idx, prev_rise_code = **8 features** (前走成績) |
| ★ SKB ★ | 拡張成績 | ❌ ★ POST-RACE LEAK 確定 (Session #38) ★ 全 10 features 完全除外 |

### 2.6 その他 (Phase 4-7+)

| code | 内容 | V15 既統合 |
|------|------|-----------|
| ★ PACI ★ | 自家ペース指数 | ✅ 統合 (paci_manken_idx, paci_goal_rank, paci_dochu_rank, paci_goal_diff, paci_jockey_exp_wr/3rd, paci_ninki_idx 他 = **11 features**) |
| Kaa, Kab | (用途未確定) | ❌ 未統合 |
| Cyb | 調教分析 | ❌ 未統合 |

---

## 3. V15 既統合 features 集計 (44 + 11 = 計 55)

### 3.1 KYI (22) — Pattern A pre-race
- jrdb_idm, jrdb_training_idx, jrdb_stable_idx, jrdb_info_idx, jrdb_composite_idx
- jrdb_upset_idx, jrdb_ten_idx_pred, jrdb_pace_idx_pred, jrdb_agari_idx_pred, jrdb_position_idx_pred
- jrdb_class_code, jrdb_rise_code, jrdb_heavy_apt, jrdb_hoof_code, jrdb_ranch_rank
- jrdb_stable_rank, jrdb_entry_days_ago, jrdb_entry_race_num, jrdb_training_arrow, jrdb_stable_eval
- jrdb_running_style, jrdb_dist_apt

### 3.2 TYB (5) — Pattern B 直前
- jrdb_paddock_idx, jrdb_odds_idx, jrdb_live_composite_idx, jrdb_body_code, jrdb_demeanor_code

### 3.3 SED (8) — 前走成績
- jrdb_prev_idm, jrdb_prev_track_bias, jrdb_prev_interference, jrdb_prev_late_start
- jrdb_prev_ten_idx, jrdb_prev_agari_idx, jrdb_prev_pace_idx, jrdb_prev_rise_code

### 3.4 KKA + extra (約 9) — CID 素点経由
- jrdb_upset_rank, jrdb_ls_rank 他

### 3.5 PACI (11) — 自家ペース指数
- paci_manken_idx, paci_goal_rank, paci_dochu_rank, paci_goal_diff
- paci_jockey_exp_wr, paci_jockey_exp_3rd, paci_ninki_idx, ...

→ ★ V15 (150 features) 中、 JRDB 由来は 44 + 11 (PACI) = **55 features (37%)** ★

---

## 4. ★ V15 未統合 (V20 候補) ★

### 4.1 ★ 外厩 (育成牧場) ★ — KYI 内に既存だが、 詳細未統合

| feature | source | 期待 corr |
|---------|--------|----------|
| jrdb_ranch_rank_detail | KYI 拡張 | 既統合 (rank A=1..E=5) |
| ★ ranch_top3_rate_recent ★ | KYI 集計 | +0.003-0.008 (★ V18 統合候補 ★) |
| ranch_history_count | KYI 集計 | +0.001-0.003 |

→ **既存 jrdb_ranch_rank** は単純 rank 1-5 のみ、 **集計 features 未統合**。

### 4.2 ★ 時系列オッズ (オッズ変動) ★ — OT/OU/OV/OW/OZ 全種未統合

| feature | source | 期待 corr |
|---------|--------|----------|
| ★ odds_change_rate_5m ★ | Oz 経由 | +0.005-0.012 |
| ★ odds_sharp_drop ★ | Oz | +0.003-0.008 |
| odds_at_n_minutes | Ot/Ou/Ov/Ow | +0.002-0.005 |
| pop_rank_history | Oz | +0.002-0.005 |

→ V18 sib_w5 と同時統合候補。

### 4.3 ★ 返し馬詳細 ★ — TYB 拡張未統合

| feature | source | 期待 corr |
|---------|--------|----------|
| return_horse_pace | TYB 拡張 | +0.002-0.005 |
| return_horse_demeanor | TYB 拡張 | +0.002-0.005 |
| return_horse_distance | TYB 拡張 | +0.001-0.003 |

### 4.4 ★ 騎手マスタ詳細 (距離別 / 馬場別) ★ — KSA 未統合

| feature | source | 期待 corr |
|---------|--------|----------|
| ★ jockey_dist_wr_master ★ | KSA | +0.003-0.008 |
| ★ jockey_surface_wr_master ★ | KSA | +0.003-0.008 |
| jockey_course_wr_master | KSA | +0.002-0.005 |

→ 既存 jockey_wr_calc / jockey_course_wr_calc / jockey_surface_wr (expanding) より集計大幅拡張。

### 4.5 KKA 詳細 (CID + 素点詳細)

| feature | source | 期待 corr |
|---------|--------|----------|
| kka_cid_detail | KKA 拡張 | +0.002-0.005 |
| kka_running_score | KKA | +0.002-0.005 |

### 4.6 CS/CSA (調教変化)

| feature | source | 期待 corr |
|---------|--------|----------|
| training_change_score | CS 経由 | +0.002-0.005 |
| training_arrow_recent | CS 経由 | +0.001-0.003 |

### 4.7 CHA (本追切詳細)

| feature | source | 期待 corr |
|---------|--------|----------|
| cha_oikiri_detail | CHA 経由 | +0.002-0.005 |
| cha_partner_score | CHA | +0.001-0.003 |

### 4.8 HJC (払戻 - jra_payouts 補完)

| feature | source | 期待効果 |
|---------|--------|---------|
| hjc_complete_payouts | HJC 経由 | jra_payouts 4/6 停止 完全 解消 |

### 4.9 期待 V20 features 追加数

| 領域 | 追加 features |
|------|--------------|
| 外厩 集計 | 2-3 |
| 時系列オッズ | 4-5 |
| 返し馬詳細 | 2-3 |
| 騎手マスタ拡張 | 3-4 |
| KKA 拡張 | 2-3 |
| CS/CHA | 3-4 |
| **合計** | **★ 16-22 features ★** |

→ V15 150 features → V20 (JRDB 拡張のみ) 166-172 features

---

## 5. JRDB の活用フロー

### 5.1 既存 fetch
- AM6:00 cron: tools/jrdb_health_check.py で 全種 DL
- 期間: 2015-2026 (11 年分、 約 600 万行)

### 5.2 既存 merge
- tools/jrdb_features.py: KYI / TYB / SED merge logic
- jrdb_kka_features.csv: KKA 別経路 merge
- features_v15_new.py: V15 拡張 (jockey_horse / transport / renovation / gaisha)

### 5.3 V20 拡張 plan
- tools/jrdb_features_v2.py 新設 (16-22 features 追加)
- 既存 merge_jrdb_train_features を継承、 expanding window で追加 features 計算
- 5/16-5/22 で実装、 5/23-5/29 で WF 検証

---

## 6. SKB POST-RACE LEAK 確定 (Session #38)

★ 重要 ★: SKB 全 10 features は POST-RACE LEAK 確定、 V20 で完全除外:
- skb_kishi_code_1/2/3 (騎手コード、 +480bp single feature contribution)
- skb_baba_code_1/2/3 (馬場コード)
- skb_kyaku_code_1/2/3 (脚質コード)
- skb_turf_hoof (芝蹄)

V20_LEAK_FEATURES = LEAK_FEATURES_A | SKB_LEAK_FEATURES (8 + 10 = **18 features 完全除外**)

---

## 7. 結論

✅ C1: 全 22 種 (実保存) 機能 確認 (BAC/CHA/CS/CYB/HJC/JO/KAA/KAB/KKA/KS/KTA/KYI/OT/OU/OV/OW/OZ/PACI/SED/SKB/TYB/UKC)
✅ C2: V15 既統合 = 55 features (KYI 22 + TYB 5 + SED 8 + KKA 9 + PACI 11)
✅ C3: V15 未統合 = ★ 16-22 features ★ (★ 外厩集計 / 時系列オッズ / 返し馬 / 騎手マスタ拡張 / KKA 詳細 / CS/CHA ★)
✅ C4: SKB 全 10 features = POST-RACE LEAK で完全除外 (V20_LEAK_FEATURES に追加)
✅ C5: V20 拡張 plan (tools/jrdb_features_v2.py 新設、 5/16-5/22 実装)

→ **V15 → V20 で JRDB 由来 +16-22 features 追加可、 期待 AUC +0.005-0.015**
→ **5/10 朝 V15 完全保証** (read-only audit、 V15 model 不変)
