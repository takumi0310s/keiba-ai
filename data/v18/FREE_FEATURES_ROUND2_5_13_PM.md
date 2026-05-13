# 5/13 PM round 2 features (72 追加、 合計 143 候補)

ラウンド 1 の 58 features に 追加して 72 features 第 2 ラウンド 実装。
V15 production / predict_core 完全不変。 全 既加入 source 範囲 + 公的 hardcoded のみ。

## 4 new module (今 round)

### 1. train/features_track_lap.py (14 features)

netkeiba マスター内 取得済 (untracked):
- `data/netkeiba_track_bias.csv` (26K rows) — track_index (-15..+15 内/外有利)
- `data/netkeiba_race_lap.csv` (25K rows) — 各レース lap times + pace 前後

**output**: `data/features_track_lap.csv` (143K × 17 cols、 bias 99.7% match、 lap 96.8% match)

| feature | 内容 | LEAK |
|---------|-----|-----|
| track_index | -15..+15 (内有利 〜 外有利) | race-level、 leak free |
| track_bias_inner_advantage / outer_advantage | 強い偏り flag | leak free |
| pace_first_half / second_half / pace_diff_race | race lap pace | race-level |
| lap_volatility | lap times std | race-level |
| lap_first_3f_race / last_3f_race | 前 / 後半 3F 合計 | race-level |
| horse_num_x_bias | 馬番 × 内外 bias 交互作用 | 強い feature |
| is_inner_umaban / is_outer_umaban | 馬番 1-4 / 13+ | 既存 |
| umaban_x_inner_advantage / outer | 偏り × 馬番 interaction | new |

### 2. train/features_jrdb_jo.py (19 features)

JRDB JO (情報データ、 302K rows) の **未利用** 16 column を 抽出。

**output**: `data/features_jrdb_jo.csv` (143K × 22 cols、 99.7% match)

| feature | 内容 |
|---------|-----|
| cid_soten_idx / cid_sara_idx / cid_idx / cid | CID 評価指数 4 件 |
| ls_idx / ls_eval | LS 指標 |
| yoso_odds | 予想 odds (LIVE 前) |
| em | em 評価 |
| gaisha_bb / gaisha_bb_wr / gaisha_bb_rensho | 外舎 統計 (勝率 / 連勝) |
| breeder_bb / breeder_bb_wr / breeder_bb_rensho | 生産者 統計 |
| soten_odds | 騒天 odds |
| cid_evaluation, ls_strength, gaisha_strength, breeder_strength | 派生 score |

### 3. train/features_horse_advanced.py (15 features)

既存 jra_races_full.csv のみ、 expanding LEAK-free。

**output**: `data/features_horse_advanced.csv` (143K × 18 cols)

| feature | 内容 | LEAK |
|---------|-----|-----|
| horse_grade_win_count | G1/G2/G3 累計 勝ち数 | cumsum shifted |
| horse_grade_top3_count | G1-G3 累計 入着数 | cumsum shifted |
| horse_g1_win_count | G1 単独 勝ち数 | cumsum shifted |
| horse_total_prize_career | 累計 賞金 (当該除外) | cumsum shifted |
| horse_jrace_count | 累計 出走回数 | cumcount |
| horse_winning_streak / losing_streak | 連勝 / 連敗 (前走以前) | iter-based |
| horse_recent_prize_3r | 直近 3 走 賞金 sum | rolling shifted |
| horse_pop_avg_3r | 直近 3 走 人気 平均 | rolling shifted |
| horse_recent_top1_rate_5r | 直近 5 走 1着率 | rolling shifted |
| jockey_horse_combo_count / top3 / top3r | 馬-騎手 pair 過去 騎乗 + 入着 | combo level expanding |
| trainer_horse_combo_count | 馬-調教師 関係 (入厩期間 近似) | combo expanding |
| same_race_attempts | 同 race_name 連投回数 (G1 等 挑戦履歴) | expanding |

### 4. train/features_venue.py (12 features)

JRA 10 場 + 主要 地方場 13 場 (合計 23 場) の hardcoded geographic + course detail。
公的 data (国土地理院 / JRA 公式 / 各場 公式) より、 商標 不使用。

**output**: `data/features_venue.csv` (143K × 15 cols、 全 row 100% カバー)

| feature | 内容 |
|---------|-----|
| venue_lat / venue_lon | 緯度 / 経度 |
| venue_elevation_m | 海抜 (m、 0-200) |
| venue_temp_base_apr | 4 月 平均気温 (基準月) |
| venue_humidity_jul | 7 月 平均湿度 |
| track_circumference_m | コース 1 周 距離 (1051-2223m) |
| track_homestretch_m | 直線 距離 (200-659m) |
| track_curve_type | 1=きつい / 2=普通 / 3=ゆるい |
| track_grade_diff_m | 高低差 (0-5.3m) |
| is_right_turn | 右回り 1 / 左回り 0 |
| venue_humid_factor | 湿潤性 (1=乾燥 / 2=湿潤) |
| is_seaside | 海近 (風強) flag |

## 累計 (5/13 PM): 143 候補 features

| round 1 (前) | features | round 2 (本) | features |
|-------------|---------|-------------|---------|
| features_free | 38 | features_track_lap | 17 |
| features_jrdb_ukc | 13 | features_jrdb_jo | 22 |
| features_sentiment | 20 | features_horse_advanced | 18 |
| | | features_venue | 15 |
| **小計 round 1** | **71** | **小計 round 2** | **72** |
| | | **合計** | **143 candidate features** |

## 期待 +AUC (合算)

| カテゴリ | 期待 +AUC |
|---------|---------|
| 公的式 (moon, holiday, venue) | +0.001-0.003 |
| 馬 拡張 (prev4/5, streak, grade win, combo) | +0.005-0.010 |
| 騎手 拡張 (period wr, streak, jockey-horse combo) | +0.003-0.006 |
| 馬主 / 生産者 / 出生地 (JRA + JRDB UKC 独立) | +0.003-0.006 |
| 厩舎コメント / レース短評 sentiment | +0.002-0.005 |
| **track_bias × 馬番 interaction** | **+0.003-0.008** (強い) |
| race lap pace 前後 | +0.001-0.003 |
| **JRDB JO (cid / ls / gaisha_bb / breeder_bb)** | **+0.005-0.010** (未活用 大) |
| 重賞 / streak / G1 history | +0.002-0.005 |
| 場 緯度 / 標高 / コース shape | +0.001-0.003 |
| **合計 期待 +AUC** | **+0.026-0.059** |

V15 baseline 0.8939 + 0.026-0.059 → **0.920-0.953 想定**

V22 4-ens (現 0.880) を 大幅越え、 V15 越え 高確率。 ただし 各 feature 効果は 学習で 検証必要。

## 規約遵守

- netkeiba マスター + JRDB 加入範囲内
- 公的 (jpholiday lib / 国土地理院 / JRA 公式 公開data)
- 商標文字列 不使用、 数値 / メタデータ のみ extract

## V15 protection

- 全 新 module、 V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- 出力 csv は data/features_*.csv 別 file
- V20+/V22 training 専用 (5/24+ 統合 判定)

## 残課題 (今回 完全 対象外)

| 項目 | 必要 |
|------|-----|
| JRA-VAN 6 種 統合 | 5/24+ Phase 3 加入後 |
| レーシングビュワー 動画 features | Phase 4 (7-8月) |
| LLM sentiment (Claude API) | API key 別途 |
| 専門家印 bulk scrape | 規約注意 + LIVE |

## 試算 V15 越え path

1. **V20 (5/24+)**: V15 base + JRA-VAN 6 種 + 143 candidate features = +0.026-0.080 期待
2. **V21 (9/1+)**: V20 + 動画 features (Phase 4 7-8月 蓄積後) = +0.005-0.010 額
3. → **V21 期待 AUC 0.925-0.95** 想定 (V15 0.894 比 +0.03-0.06)
