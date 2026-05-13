# 無料 + 既加入 source 範囲内 feature 拡張 (5/13 PM)

V20+/V22 学習で merge する 候補 feature を 4 module 一気 実装。
V15 production / predict_core / daily_predict 完全不変。

## 1. train/features_free.py (38 features)

外部 source 不要、 公的式 + 既存 jra_races_full.csv で計算。

**output**: `data/features_free.csv` (143,122 rows for 2023+)

| カテゴリ | features | 工数 | 期待 +AUC |
|---------|---------|-----|---------|
| 月齢 | moon_phase, is_full_moon, is_new_moon | Conway 式、 license 制限なし | +0.000-0.002 |
| 祝日 | is_holiday, consec_holiday_days, is_long_holiday, weekday, is_weekend | jpholiday lib | +0.001-0.003 |
| 馬年齢 | horse_age_months, horse_birth_month, horse_birth_quarter | 生年月日 + race date | +0.001-0.003 |
| 馬連戦 | horse_same_course_count, days_since_prev, is_short_rest, is_very_short_rest | expanding | +0.002-0.004 |
| 騎手 期間別 wr | jockey_streak_win, jockey_recent_wr_60r/180r, top3r_60r/180r | expanding rolling | +0.002-0.005 |
| 馬主 / 生産者 | owner_top3r, owner_wr, breeder_top3r, breeder_wr | expanding Bayesian | +0.002-0.005 |
| 前走 拡張 | prev4_finish/last3f/pop, prev5_*, prev_same_course, prev_same_distance, prev_dist_diff | lag 拡張 | +0.001-0.003 |
| race 位置 | race_num_cat, is_main_race | 既存 | +0.0-0.001 |

**LEAK-free 保証**: 全 expanding (cumsum - current_row) で 当該レース 除外。 Bayesian alpha 30-50。

## 2. train/features_jrdb_ukc.py (10 features)

既加入 JRDB UKC csv (36,939 馬登録) を horse_id ↔ blood_num で merge。 86.8% マッチ。

**output**: `data/features_jrdb_ukc.csv` (143,122 rows × 13 cols)

| feature | 内容 | 期待 +AUC |
|---------|-----|---------|
| owner_code_top3r_jrdb | JRDB 馬主 code expanding top3r | +0.001-0.002 |
| owner_code_wr_jrdb | JRDB 馬主 code expanding wr | +0.001-0.002 |
| birthplace_top3r | 出生地 (北海道日高 / 胆振 等) expanding | +0.001-0.003 |
| birthplace_wr | 出生地 win 率 | +0.000-0.001 |
| father_code_top3r | 種牡馬 code 単独 expanding | +0.001-0.002 |
| bms_code_top3r | 母父 code expanding | +0.000-0.001 |
| age_gap_father, age_gap_bms | 種牡馬 / 母父 世代 gap | +0.000-0.001 |
| father_birth_year_ohash, bms_birth_year_ohash | 族系 cohort | +0.000-0.001 |

## 3. train/features_sentiment.py (5+5 features)

**output**: `data/features_stable_comment_sentiment.csv` (130,316 rows × 10 cols)、
`data/features_race_review_sentiment.csv` (277,466 rows × 10 cols)

simple keyword dictionary (POSITIVE 27 + NEGATIVE 27 + CRITICAL 6)、 LLM 不要。

| feature | 内容 |
|---------|-----|
| stable_comment_positive_count | 厩舎コメント 中 ポジティブ語 数 |
| stable_comment_negative_count | 同 ネガティブ語 数 |
| stable_comment_critical_neg | 取消 / 骨折 / 故障 等 重大 ネガ語 (-2 重み) |
| stable_comment_net_score | pos - neg - 2*crit |
| race_review_* | 同上 5 column 構造 (前走 review 用) |

**期待 +AUC**: +0.002-0.005 (実行時 直前情報)

## 4. 即実装 推定 効果 + 5/24+ V20 投入 想定

| module | features | 即 利用可能 (LEAK-free) | 期待 +AUC 合計 |
|--------|---------|----------------------|---------------|
| features_free | 38 | yes | +0.005-0.020 |
| features_jrdb_ukc | 10 | yes | +0.003-0.008 |
| features_sentiment | 10 | yes (lookups は historical data) | +0.002-0.005 |
| **合計** | **58 新 features** | — | **+0.010-0.033 AUC** |

V15 baseline 0.8939 + 0.010-0.033 → 0.904-0.927 想定 (V22 単独 0.88 を 越え、 V21 video features 待たず V15 superseding 可能性)。

## 5. V20+/V22 学習 で merge する 手順 (5/24+)

```python
# train/train_v22_with_extras.py (今後 5/24+ 実装)
df = pd.read_csv('data/jra_races_full.csv', low_memory=False)
df_free = pd.read_csv('data/features_free.csv')
df_ukc = pd.read_csv('data/features_jrdb_ukc.csv')
df_sc = pd.read_csv('data/features_stable_comment_sentiment.csv')

# merge keys: race_id + horse_id + umaban
df = df.merge(df_free, on=['race_id', 'horse_id', 'umaban'], how='left')
df = df.merge(df_ukc, on=['race_id', 'horse_id', 'umaban'], how='left')
df = df.merge(df_sc[['race_id', 'umaban', 'stable_comment_net_score']],
              on=['race_id', 'umaban'], how='left')
# → V22 4-ens 学習 with extras
```

## 6. V15 production 完全保護 (本日も遵守)

- features_*.py は 全 新 module、 V15 predict_core / app.py 完全不変
- 出力 csv は data/ 下、 V15 学習 lookups から 分離
- V20+/V22 training 専用 (5/24+ V20 投入時 統合)

## 7. 残課題 (今回 対象外)

| 内容 | 状態 | 工数 |
|------|-----|------|
| **専門家印 bulk scrape** | data 13 件のみ (取得 必要) | 1-2 日 (LIVE 取得 path、 規約注意) |
| **騎手コメント LLM 数値化** | text data 少、 LLM API 別途 | 1 日 (Claude API) |
| **JRDB CHA / JO 全 column 監査** | 既 merge 部分あり | 半日 |
| **JRA-VAN 6 種 統合** | 5/24+ Phase 3 | 2-3 日 |
| **動画 features (Phase 4)** | 7-8 月 | 1 週間 |

## 8. test 結果

- 全 module syntax check OK
- features_free build 動作 OK (143K × 38)
- features_jrdb_ukc build 動作 OK (143K × 13、 UKC マッチ 86.8%)
- features_sentiment build 動作 OK (130K + 277K)
- V15 1 race 予測 動作 OK (フローラS、 影響なし 確認)
- predict_core import w/ -W error::FutureWarning OK

## 9. 規約 / 商標 遵守

全 source は 既加入 (netkeiba マスター / JRDB) + 公的 (jpholiday / 天文式)。 商標 文字列 表示 不使用、 数値 / メタデータ のみ extract。
