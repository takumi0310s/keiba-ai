# V20 features manifest FINAL (5/11 marathon 全 verify、 寝るまで edition)

## ★★★★ TOP signals (Q5 vs Q1 / 全条件 検証済)

### Mega signals (+15pt 以上、 5/11 marathon 発見)
| # | feature | signal | source |
|---|---------|--------|--------|
| 1 | **horse_recent5_top3** | **+24.0pt** | hot_streak_features.csv |
| 2 | **jockey_recent30_top3** | **+24.0pt** | hot_streak_features.csv |
| 3 | **trainer_recent30_top3** | **+15.2pt** | hot_streak_features.csv |

### Strong signals (+10pt 以上)
| # | feature | signal | source |
|---|---------|--------|--------|
| 4 | class_down (降級) | +12.5pt | event_effect_features.csv |
| 5 | pace_career_burst_mean (差し力) | +10.1pt | pace_features_expanding.csv |
| 6 | pace_career_change_1to4_mean | +10pt | 同上 |

### Mid signals (+5-10pt または expanding)
| # | feature | signal | source |
|---|---------|--------|--------|
| 7 | class_down_top3_rate_exp | strong | event_effect_features.csv |
| 8 | jockey_change_top3_rate_exp | importance 9,760 | event_effect_features.csv |
| 9 | trainer_change_top3_rate_exp | 2,260 | 同上 |
| 10 | class_up_top3_rate_exp | 2,936 | 同上 |
| 11 | pace_career_relative_4cor_mean | 7,876 | pace_features_expanding |
| 12 | pace_recent5_burst_mean | 1,562 | 同上 |
| 13 | pace_recent5_change_mean | 765 | 同上 |
| 14 | sire_class_down_boost_exp | NEW | sire_class_down_features.csv |

### Negative signals (低 ほど 良い)
| # | feature | signal | source |
|---|---------|--------|--------|
| 15 | surface_change | -9.3pt | distance_surface_change_features.csv |
| 16 | dirt_to_turf | -10.4pt | 同上 |
| 17 | turf_to_dirt | -7.0pt | 同上 |
| 18 | very_long_layoff (6m+) | -7.4pt | layoff_features.csv |
| 19 | long_layoff (3m+) | -4.6pt | 同上 |
| 20 | trainer_change | -3.0〜-7.7pt | event_effect_features.csv |

## 黄金 pattern (3-way / 4-way interaction)

### 3-way: top3 rate **43.8%** (baseline 22.8%、 +21pt)
- class_down = 1
- jockey_change = 0
- pace_career_burst_mean = Q5 (上位 20%)

### 4-way (sire 付き): top3 rate **50-64%**
- 上記 3-way 条件
- + 父馬 = キズナ (64.5%) / ミッキーアイル (58.3%) / サートゥルナーリア (58.3%) /
  ドレフォン (57.1%) / キタサンブラック (53.8%) etc.

### 単勝 ROI (popularity 推定)
- 黄金 pattern 全体: **180.4%** (全馬 80% から +100pt)
- 2-3 人気 sweet spot: **192.0%**
- 8 人気+: **245.0%** (穴で爆発)

## V20 学習で 必須 採用 features (合計 約 40 件)

### 1. 既存 V15 features (~150 件) を 維持

### 2. NEW 強 signals (今夜検証済、 14 件)
- class_down + 関連 5 features
- pace_career_burst + 関連 4 features
- hot_streak 3 features (横軸最強)
- sire_class_down boost 4 features
- distance/surface change 関連 10 features
- layoff 関連 7 features
- 動画 AI 関連 38 features (paddock 蓄積後 V21 で)

### 3. 明示 interaction (V20 で explicit feature 化 推奨)
- `class_down * (burst >= Q5)` (43.8% top3 pattern)
- `class_down * (1 - jockey_change)` (同騎手で降級 boost)
- `class_down * (1 - trainer_change)` (同厩舎で降級 boost)

### 4. 必須 除外 LEAK features
- V15 LEAK_FEATURES_A (8 件)
- SKB_LEAK_FEATURES (10 件、 Session #38)
- pass1-4, agari_3f 生 (POST-RACE)

## 期待効果 (V20 投入時)

| 指標 | V15 baseline | V20 想定 | 改善 |
|------|-------------|---------|------|
| WF AUC (top3) | 0.8939 | **0.900-0.920** | +0.006-0.026 |
| 実 ROI (戦略⑦込み) | 140% | **150-170%** | +10-30pt |
| 月利 想定 | +¥28K | **+¥50-100K** | +¥22-72K |

※ AUC +0.114 が hot_streak 等 含めた全 NEW features の単独 effect (~30K rows baseline)
V15 既存 features と overlap で 実 incremental は +0.005-0.015 想定。
ただし hot_streak は V15 既存になく 純増、 incremental は大き目期待。

## 5/24+ V20 投入 plan (Phase 25)

### 5/18-22 V20 学習 data 構築
1. `tools/v21_training_data_builder.py` で merge data 生成
2. + hot_streak_features.csv merge 追加
3. + layoff_features.csv merge 追加
4. + distance_surface_change_features.csv merge 追加
5. + sire_class_down_features.csv merge 追加

### 5/23 V20 GO/no-go 判定
- WF AUC ≥ 0.900
- 全条件 ROI ≥ 100%
- LEAK 監査 PASS

### 5/24 V20 段階投入 (GO の場合)
- V15 並行運用 1 ヶ月
- V20 投資額 上限 5,000 円/日
- shadow log 同時並行

## V15 投資保護 (絶対遵守、 5/11 marathon 終始 厳守)

predict_core.py / daily_predict.py / app.py / V15 model `.pkl.gz` 一切 不変。
全 Phase 21D-24 / 35+ tools / 30+ commits は post-process / 検証 / 分析 / V20 準備。
5/17 開催 = V15 案 B 改 + 戦略⑦ 単独継続 確定。

## まとめ

今夜 1 session で:
- **35+ tools 実装**
- **31+ commits / 約 8,000 行**
- **AUC +0.114 (~30K rows baseline 0.626→0.740) 実証**
- **+24pt 大 signals (hot_streak) を 3 個 発見**
- **黄金 pattern 単勝 ROI 推定 180%**
- **キズナ × 黄金 pattern で top3 64.5%**
- **V20 投入 features 約 40 件 確定**

V15 production 安全保護下で **未開拓の signal 多数発見**、 V20 投入で 月利 +¥50-100K の見込み。
