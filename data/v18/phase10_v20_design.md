# Phase 10 E: 4 source 統合 V20 設計 + 1 週間限界 plan (5/10)

> Session #87 (2026-05-10 夜) Phase 10 E 領域
> 対象: ★ 全 4 source (DataLab + RV + JRDB + netkeiba マスター) 統合 V20 ★
> 趣旨: V15 (150 features) → V20 (250+ features) 設計、 期待 AUC 0.91-0.93

---

## 1. 4 source 加入状況 サマリ

| source | 月額 | 加入日 | V15 既統合 features | 未統合 候補 |
|--------|------|-------|--------------------|------------|
| ✅ JRA-VAN DataLab | ¥2,090 | 2026-05-07 | 11/26 種 (RA/SE/HR/UM/TM/O1/WC/WE/KS/SK 一部) | 5-10 features (O5/BN/HS/WF/BR) |
| ✅ JRA レーシングビュアー | ¥550 | 2026-05-10 | 0 features (動画) | 12 features (V21 で統合、 9/2 投入候補) |
| ✅ JRDB Advance | ¥2,880 | 既加入 | 55 features (KYI 22 + TYB 5 + SED 8 + KKA 9 + PACI 11) | 16-22 features (外厩集計 / 時系列オッズ / 騎手マスタ拡張 等) |
| ✅ netkeiba マスター | ¥4,980 | 2024-12 | 6 features (master_index / track_bias 部分 / lap 部分 / siblings / speed_index 等) | 30-40 features (AI 展開 / 波乱度 / 個別ラップ / 走行距離 / レース相性) |
| **合計** | **¥10,500/月** | — | **★ V15 150 features (既統合) ★** | **★ +51-72 features 追加可 ★** |

→ V20 (動画なし、 V15 + 4 source 拡張): **★ 約 200-220 features ★**
→ V21 (V20 + 動画 features 12): **★ 約 215-235 features ★**
→ V22 (V21 + RL 投票最適化): features 数同じ、 投票配分のみ変化

---

## 2. V20 全 features 構成 (★ V15 150 → V20 200-220 ★)

### 2.1 V15 既統合 (150 features) - そのまま継承

- 基本 / レース条件: 23
- 前走 / レビュー: 12
- 通算成績 / 集計: 8
- 騎手: 8
- 厩舎 / 調教師: 3
- 調教 (netkeiba 経由): 12
- 血統 (父/母父): 6
- タイム指数 (netkeiba): 4
- JRDB KYI / TYB / SED / KKA: 44
- JRDB PACI: 11
- オッズ / 人気 (Stage 2): 6
- その他 / 派生: 13
- **計 150 features**

### 2.2 V20 新規 (+51-72 features)

#### A. JV-Link DataLab (+5-10 features)
- jv_o5_trio_odds_open / final (三連複 オッズ)
- jv_o3_umatan_odds, jv_o4_wide_odds
- jv_bn_owner_top3_3y (馬主 3 年 top3)
- jv_hs_breeder_top3_5y (生産者 5 年 top3)
- jv_wf_win5_appearance_3y (WIN5 出走数)
- jv_br_dam_offspring_count (繁殖牝馬産駒数)
- jv_hr_complete_payout (jra_payouts 4/6 停止 解消)

#### B. JRDB 拡張 (+16-22 features)
- jrdb_ranch_top3_rate_recent (外厩 直近 top3)
- jrdb_ranch_history_count
- jrdb_odds_change_rate_5m (時系列オッズ変動)
- jrdb_odds_sharp_drop
- jrdb_odds_at_n_minutes (5 段階)
- jrdb_pop_rank_history
- jrdb_return_horse_pace (返し馬)
- jrdb_return_horse_demeanor
- jrdb_return_horse_distance
- jrdb_jockey_dist_wr_master (騎手マスタ拡張)
- jrdb_jockey_surface_wr_master
- jrdb_jockey_course_wr_master
- jrdb_kka_cid_detail
- jrdb_kka_running_score
- jrdb_training_change_score (CS)
- jrdb_training_arrow_recent
- jrdb_cha_oikiri_detail (本追切詳細)
- jrdb_cha_partner_score

#### C. netkeiba マスターコース (+30-40 features)
- nk_ai_position_pass1/2/3/4 (AI 通過順)
- nk_ai_agari_pred (上がり 3F 予想)
- nk_ai_upset_score (波乱度 0-1)
- nk_ai_predict_time_total (走破タイム予想)
- nk_ai_predict_lap_first3f / last3f
- nk_ai_grade (50 段階)
- nk_track_bias_inner / outer / center (枠別有利度)
- nk_track_bias_pace_speed
- nk_individual_lap_avg / std
- nk_individual_lap_first3f / last3f
- nk_running_distance_total (実走行距離)
- nk_position_loss
- nk_race_lap_pattern_enc
- nk_combinations_top_picks (集合知 上位)
- nk_odds_sensor_alert
- nk_horse_compatibility

→ ★ V20 = 150 (既存) + 51-72 (新規) = **約 200-220 features** ★

### 2.3 V21 動画 features (+12 features、 V21 投入時)
- video_paddock_body_score, sweat_level, hoof_condition, hindleg_drive, calmness_score
- video_patrol_furi_count, route_efficiency, block_detection, pace_change
- video_training_stride_length, gait_symmetry, finish_speed

→ V21 = V20 + 12 = **約 212-232 features**

---

## 3. ★ 期待 AUC + ROI 試算 ★

### 3.1 期待 AUC

| Model | AUC (BT WF) | AUC 改善 | 主寄与 |
|-------|------------|---------|--------|
| V15 (現行) | 0.886 (3-fold) / 0.8939 (公称) | baseline | — |
| V20 case A: JRDB 拡張のみ | 0.890-0.895 | +0.005-0.010 | jrdb +16-22 features |
| V20 case B: V20 case A + DataLab | 0.892-0.898 | +0.006-0.012 | + jv +5-10 features |
| ★ V20 case C: V20 case B + netkeiba マスター ★ | **0.910-0.925** | **+0.020-0.040** | + nk +30-40 features (AI 展開 / 個別ラップ corr 高) |
| V21 (V20 + 動画) | 0.920-0.940 | +0.030-0.054 | + 動画 +12 features |
| V22 (V21 + RL 投票) | 同上 | (AUC 同じ) | RL で ROI 改善 |

### 3.2 期待 ROI 試算

| Model | 期待 ROI (戦略⑦込み) | 月利 (700円×30R 想定) |
|-------|---------------------|------------------------|
| V15 (現行) | 119.2% (実績) / 140% (戦略⑦込) | 約 +¥6,000-13,000 |
| V20 case C (4 source 統合) | **150-160%** | **約 +¥10,500-13,000** |
| V21 (V20 + 動画) | 155-170% | **約 +¥12,000-15,000** |
| V22 (V21 + RL) | **165-180%** | **約 +¥14,000-17,000** |

→ 月額 ¥10,500 (4 source) は V20 月利 +¥10K-13K で **回収 OK**

---

## 4. ★ 1 週間 限界点 plan (5/11-5/17) ★

### Day 1 (5/11 日) - 4 source full audit (本日完了済 + 残作業)
- ✅ Phase 10 audit (DataLab / RV / JRDB / netkeiba マスター 完了)
- 5/11 朝: V15 朝予測 + 結果照合
- 5/11 夜: 5/16 V18 sib_w5 投入候補 doc 整理
- 残: ★ AI 波乱度 / 走行距離 PC scrape 経路 探索 ★
- 残: ★ JV-Link 32-bit Python venv 動作再確認 ★

### Day 2 (5/12 月) - JRDB 未統合 + netkeiba マスター 大量取得
- 朝: V15 平日 monitoring
- 日中:
  - tools/jrdb_features_v2.py 新設 (16-22 features)
  - 外厩集計 / 時系列オッズ / 騎手マスタ拡張 implementation
  - netkeiba マスター: AI 波乱度 + 走行距離 大量取得 (DELAY=10 conservative)
- 夜: WF 検証 用の data prep

### Day 3 (5/13 火) - netkeiba マスター 残機能 取得 + V20 学習 data 構築
- tools/scrape_master_course.py 拡張 (走行距離 / レース相性度)
- AI 予想タイム 大量取得 (現状 17 行 → 50K+ 行 目標)
- V20 学習 data: V15 base + 4 source 全 features merge
- WF 6-fold 用 data prep 完了

### Day 4 (5/14 水) - V20 ensemble 学習 (4-model)
- LGB + XGB + FT-Transformer + IntraRace Attention
- WF 6-fold (2020-2025)
- 期待 AUC 0.910-0.925
- ★ 6-fold AUC 全て 0.90 超える かつ V15 と同等以上の場合のみ採用 ★

### Day 5 (5/15 木) - V20 paper trade test
- 5/14 学習結果 で 5/15 paper trade
- 投資判断 only、 実投票なし
- 想定 ROI 150-160% 達成確認

### Day 6 (5/16 金) - JRA-VAN RV trial 開始 + V20 paper trade 継続
- ★ JRA-VAN RV trial 開始 (5/15 → 5/16 に shift) ★
- 重賞調教動画 視聴 + features 抽出 試行
- V20 paper trade 5/16 平日 monitoring

### Day 7 (5/17 土) - V20 paper + V15 本番 並行運用
- 朝: V15 本番 投票 (案 B 改 strict)
- 同時に V20 paper 予測 (実投票なし)
- 夜: V15 vs V20 比較 → 5/24+ V20 投入候補 GO/no-go 判定

---

## 5. ★ V15 投資保護 (絶対遵守) ★

### 5.1 不変 (絶対)
- predict_core / daily_predict / app.py: 不変
- V15 model file: 不変 (md5 fingerprint check)
- schtask: 不変 (DailyPremiumScrape / DailyPredict / RaceAutoNotify / DailyResults)
- 戦略⑦ (race_auto_notify.py): 不変
- 案 B 改 戦略 (5/9 投票 logic): 不変

### 5.2 V20 paper trade 専用環境
- 別 dir: `data/v20_paper/` 出力
- 別 process: 既存 daily_predict と並行 launch
- 既存 model file 読み書き しない

### 5.3 撤退条件 (絶対遵守)
- 累計 -¥50,000 で撤退、 V20 計画 中止
- 5/17 V20 paper ROI < 130% で V20 投入延期 (3 週間 検証延長)
- 4 source 規約違反 alert で 該当 source 利用停止

---

## 6. ★ V20 投入 timeline (1 ヶ月前倒し) ★

| 日 | event | 状況 |
|---|-------|------|
| ★ 5/11-5/15 ★ | V20 features 追加 + 学習 | ★ 1 週間限界 plan ★ |
| 5/16-5/22 | V20 paper trade 1 週間 | (旧 plan 5/16-5/22) |
| 5/23-5/29 | V20 LIVE retro | (Session #44 plan) |
| 5/30-6/7 | V20 paper trading 継続 + bug fix | (Session #44 plan) |
| ★ 6/8 (日) ★ | ★ V20 投入候補 GO/no-go 判定 ★ | (旧 plan 7/1 から **1 ヶ月前倒し** 維持) |

---

## 7. ★ 結論 ★

✅ E1: V15 (150) → V20 (200-220) features 設計確定
✅ E2: 4 source 統合 期待 AUC ★ 0.910-0.925 ★ (V15 0.886 から +0.020-0.040)
✅ E3: 期待 ROI 150-160% (戦略⑦込み)、 月額 ¥10,500 回収 OK
✅ E4: 1 週間限界 plan (Day 1-7、 5/11-5/17) 確定
✅ E5: V15 投資保護 完全 (paper trade 専用 dir、 撤退条件 -¥50,000)
✅ E6: V20 投入 6/8 (1 ヶ月前倒し) 維持

→ **★ V20 = 4 source 完全統合 ensemble、 期待 AUC 0.91-0.93、 5/17 paper trade、 6/8 投入候補 ★**
→ **★ 5/10 朝 V15 完全保証 ★** (read-only audit、 V15 model 不変)
