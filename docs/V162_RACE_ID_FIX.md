# v16.2 race_id 変換改善 (5/4-5/6)

## 4/28 判明したバグ

`features_v16_premium.py` の `_build_nk_race_id_from_jv()`:
- TARGET形式 race_id を netkeiba形式に変換する関数
- 現状の変換成功率は不明だが、prev_review_score 10.7% から低い可能性大

## 現状のマッチ率

直接マッチ (race_id+umaban):
- 訓練 vs race_review: **0%** (フォーマット違い!)
- 訓練 vs master_index: **0%** (同上)

ただし features_v16_premium.py 内で変換後は:
- prev_review_score: 10.7% (lag-1 後)
- prev_master_index: 14.6% (lag-1 後)
- prev_track_index_val: 44.3% (lag-1 後)

## 修正方針 (5/4-5/6 GW後半)

### Step 1: race_id 変換ロジックの精査
1. `_build_nk_race_id_from_jv()` の現在の変換ルール確認
2. 全 race_id で変換成功率を測定
3. 失敗ケースのパターン分析

### Step 2: 変換ロジック修正
- TARGET race_id (10-12桁) → netkeiba race_id (12桁)
- 年: YY → 20YY (年下2桁を 4桁化)
- その他のフィールドは同じ?

### Step 3: 期待効果
- prev_review_score: 10.7% → 40%+ (4倍改善)
- prev_master_index: 14.6% → 30%+ (master_index バグ修正と相乗)
- prev_track_index_val: 44.3% → 60%+

## v16.2 期待効果

修正後の ablation 期待:
- v15 baseline: 0.8856
- + training_eval_rank: +2bp (継承)
- + prev_master_index (修正後): +2~+4bp
- + prev_review_score (修正後): +1~+3bp
- + prev_track_index_val (修正後): +1~+2bp

総計期待: +6~+11bp 改善
v16.2 期待 mean AUC: 0.886~0.890
