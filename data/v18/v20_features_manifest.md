# V20 features manifest (5/24+ 学習用、 verify 済 signal 集約)

## ★★★ Confirmed strong signals (LEAK-free、 5/11 検証済)

### 1. class_down (降級 effect) - #1 importance
- **importance: 67,372** (full LGB model 中 #1)
- 統計: top3 rate 0.325 (class_down=1) vs 0.200 (=0) = **+12.5pt**
- 動作確認: 532,004 rows、 n=96,114 (18.1%)
- 算出: `tools/build_event_effect_features.py`
- LEAK: 当該 race の class_code は事前確定 → safe

### 2. class_down_top3_rate_exp (降級時の career top3 rate)
- **importance: 4,486** (#9 in top 15)
- expanding window で safe
- 算出: 同上 (event_effect_features.csv)

### 3. pace_career_burst_mean (career 差し力)
- top3 rate by quintile:
  - Q1 (低): 17.8%
  - Q5 (高): **27.9%** (+10.1pt)
- 動作確認: 94,249 rows、 n=77,426 non-null (82%)
- 算出: `tools/build_pace_features_expanding.py`
- LEAK: expanding (当該 race 除外) で safe

### 4. jockey_change_top3_rate_exp (騎手乗替時の career top3 rate)
- **importance: 8,178** (#7)
- expanding で safe

### 5. jockey_change (騎手乗替 binary)
- **importance: 7,047** (#8)
- 当該 race の騎手は事前確定 → safe

### 6. trainer_change + _top3_rate_exp
- importance 3,231 + 2,554 (#12, #13)
- 同様 safe

## ★★ Medium signals

### 7. class_change (升降級 binary)
- importance: 846
- single binary は弱、 但し class_up/down と組合せで強

### 8. pace_career_change_1to4_mean (career 前進/後退)
- ↑ pace_features_expanding に含む

### 9. pace_career_relative_4cor_mean (4角 相対位置)
- ↑ 同上

### 10. pace_recent5_burst_mean / change_mean
- recent 5 走 平均、 trend reflection

## ★ Weak signals (個別では 0 importance、 interaction 候補)

### remarks 短評 categorical (rmk_*)
- 全 9 features、 LGB importance 全て 0
- 単独では V12 既に 不採用 確定
- ただし interaction (rmk_delay × class_down 等) で再検証 余地あり

## 動画 AI features (Phase 22 で確認、 paddock 蓄積後 V21 で 採用)

### gait features (20 種、 video_ai_gait_features.py)
- aspect_mean / std / range
- area_mean / std
- conf_mean / std
- motion_speed_mean / std / max
- aspect_change_mean (gait 周期)
- ...

### body_condition features (18 種、 video_ai_body_condition.py)
- coat_brightness / saturation / contrast (毛色 状態)
- body_aspect / compactness
- condition_score (heuristic 合成)
- 各 mean + std (over frames)

## V20 学習で 確実に追加すべき features (まとめ)

### 必須追加 (LEAK-free 検証済)
1. ✅ class_down
2. ✅ class_down_top3_rate_exp
3. ✅ class_up
4. ✅ class_up_top3_rate_exp
5. ✅ class_change
6. ✅ jockey_change
7. ✅ jockey_change_top3_rate_exp
8. ✅ trainer_change
9. ✅ trainer_change_top3_rate_exp
10. ✅ pace_career_burst_mean
11. ✅ pace_career_change_1to4_mean
12. ✅ pace_career_relative_4cor_mean
13. ✅ pace_recent5_burst_mean
14. ✅ pace_recent5_change_mean

合計 **14 features 追加** (V15 既存 124 features + 14 = 138 features)

### 候補 (interaction で再検証)
- rmk_delay × class_down (出遅れ + 降級 = ?)
- rmk_trouble × jockey_change (前走不利 + 騎手変更 = ?)

### V21 追加 (paddock 蓄積後)
- gait 20 + body_condition 18 = 38 video features
- multi-horse tracking で race 動画 features も可能

## 必須除外 features (LEAK 厳禁)

### V15 LEAK_FEATURES_A (8 features)
- odds_log, horse_weight, condition_enc
- weight_change*, weight_cat*, cond_surface

### V20 SKB_LEAK_FEATURES (10 features、 Session #38 確定)
- skb_kishi_code_1/2/3
- skb_baba_code_1/2/3
- skb_kyaku_code_1/2/3
- skb_turf_hoof

### 注意要 (生 pace features)
- pass1-4, agari_3f, run_time (当該 race の値)
- final_burst (post-race)
- pos_change_1to4 (post-race)
- これらは pace_features.csv にあるが、 **expanding 版を使うこと**

## sib_*_exp 修正版 (Session #39 C 実装済)
- sib_top3_rate_exp など、 修正済 (LEAK 除去 + 信号残存)
- corr_target 0.169 (旧 0.294 から リーク除去後)

## 期待効果 (V20 投入時)

| 指標 | V15 baseline | V20 想定 | 改善 |
|------|-------------|---------|------|
| WF AUC (top3) | 0.8939 | 0.895-0.905 | +0.001-0.011 |
| 実 ROI | 戦略⑦ 140% | 戦略⑦ 145-150% | +5-10pt |
| 月利 想定 | +¥28K | +¥35-50K | +¥7-22K |

※ V21 で動画 features 採用 後は さらに +10-20% ROI 想定 (paddock 数千 race 蓄積後)

## 学習 procedure (5/24+ user task)

1. `python tools/v21_training_data_builder.py --year-from 2020 --year-to 2025` → 基盤 CSV
2. `train/train_v20_*.py` 実装 (本 manifest 参照、 V15 train script を base)
3. LEAK 監査 22 項目 通す
4. WF AUC ≥ 0.880 確認
5. 6/8 GO/no-go 判定

## V15 投資保護 厳守

- V15 model file (.pkl.gz) は **完全 freeze**
- V20 output は `keiba_model_v20_*.pkl.gz` (新 file 名)
- 1 ヶ月並行運用 (6/24+) で 比較確認後 V15 archive 判定
