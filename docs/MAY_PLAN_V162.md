# 5月の計画: v16.2 への道

## 4/28 判明事項

### 充填率と効果の関係

| 特徴量 | 充填率(全) | 訓練期間 | ablation | 真の判定 |
|--------|-----------|---------|----------|---------|
| training_eval_rank | 53.6% | 高 | +2bp ✅ | **採用** (v16.1) |
| top_popularity_reliability | 100% | 100% | -11bp | 真の効果なし |
| prev_track_index_val | 44.3% | ? | -3bp | 要検証 |
| upset_level_val | 23.7% | ? | -1bp | 要検証 |
| prev_master_index | 14.6% | ? | -13bp | カバレッジ不足? |
| prev_review_score | 10.7% | ? | -2bp | カバレッジ不足? |

## 5月優先タスク

### 5/2-5/3 GW初日・2日目 (本番運用)
- v16.1 で 36R 自動運用
- 戦略⑦v1 で 4R 除外
- 期待 ROI 109.6%
- 実績データ収集

### 5/4-5/6 GW後半 (バグ修正と再ablation)

#### 1. master_index 取得バグ修正 (最優先)
- tools/scrape_master_index.py の Already scraped 判定修正
- year フィルタを追加
- 2020-2023 を再取得試行
- 期待: 充填率 14.6% → 30-40%

#### 2. prev_review_score の充填率向上
- netkeiba_race_review.csv は 277,466行ある
- でも prev_review_score (lag-1) は 10.7%
- マッチングロジック確認
- 改善余地大きい

#### 3. ablation 再実行 (master_index 修正後)
- v16 全特徴量で再評価
- カバレッジ問題が原因だった特徴量を再発掘
- v16.2 候補確定

### 5/7 以降 (v16.2 学習)
- 採用された特徴量で v16.2 構成
- 期待 mean AUC: 0.886 (+4-5bp 改善)
- 5/9 GW週末で v16.2 デビュー

## 重要な学び

### top_popularity_reliability の教訓
- 100% 充填でも -11bp 悪化
- → カバレッジが全てではない
- → 真の予測価値が重要

### training_eval_rank の成功
- 53.6% 充填で +2bp 改善
- → 充填率より「有用なシグナル」が重要
- → 採用クリア
