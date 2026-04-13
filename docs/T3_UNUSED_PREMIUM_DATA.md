# T3: 未使用プレミアムデータ調査レポート

Date: 2026-04-13
Baseline: v15 / 150特徴量 / WF AUC 0.8856

## netkeiba プレミアムCSV 使用状況

### 現在モデル使用中（v15 特徴量と紐づく）

| CSV | 用途 | モデル特徴量 |
|-----|------|--------------|
| `netkeiba_training_times.csv` | 追切タイム | training_time_filled, wood_best_4f_filled, sakaro_*, time_1f_last_filled |
| `netkeiba_speed_index.csv` | タイム指数 | index_max_filled, index_run1_filled, index_avg5_filled |
| `netkeiba_stable_comments.csv` | 厩舎コメント | stable_comment_score |
| `netkeiba_siblings.csv` | 母産駒成績 | （表示のみ、モデル未使用） |

### 取得済みだが**モデル未使用**のCSV

| CSV | 行数 | 鮮度 | 特徴量候補 |
|-----|------|------|-----------|
| `netkeiba_upset_level.csv` | 19,378 | 2020-2024 | **★1 波乱度Lv1-5 + 上位人気信頼度** |
| `netkeiba_race_review.csv` | 277,467 | 2020-2025全年 | **★2 前走不利短評 review_score** |
| `netkeiba_training_eval.csv` | 95,066 | 2024-2025 | **★3 追切評価ランク+コメント+短評スコア** |
| `netkeiba_shinba_eval.csv` | 7,999 | 2024-2025 | stable_eval/training_rank/score |
| `netkeiba_master_index.csv` | 7,502 | 2025のみ | **★4 3分解指数 time/start/chase/agari_index** |
| `netkeiba_track_bias.csv` | 9,436 | 2020-2022+2025 | **★5 馬場指数 track_index + trackバイアステキスト** |
| `netkeiba_race_lap.csv` | 9,097 | 2020-2022+2025 | pace_first_half / pace_second_half（pciと類似） |
| `netkeiba_ai_position.csv` | 67,953 | 2025 | AI展開予測ポジション（当日8時までに配信） |
| `netkeiba_ai_opinion.csv` | 4,930 | 2025 | AIペース予想 + opinion_text |
| `netkeiba_ana_best.csv` | 12,381 | 2025 | 能力/上昇度ピックアップ |
| `netkeiba_race_analysis.csv` | 52,765 | 2025 | 馬単位コメント/スコア |
| `netkeiba_data_analysis.csv` | 1,956 | 2025 | レース傾向分析 |
| `netkeiba_track_index.csv` | 6,946 | 2025 | 馬場指数（track_biasと重複） |
| `netkeiba_race_tendency.csv` | 13 | ほぼ空 | — |
| `netkeiba_ai_predict_times.csv` | 17 | ほぼ空 | — |

## 上位5新特徴量候補（AM8:00取得可能性 × AUC改善余地）

### 採用基準
- AM8:00 取得可能（前日/当日朝の配信データ）
- 全年カバレッジ可能 or 将来的に埋められる
- リークフリー（前走データ・前日情報）
- 既存 150 特徴量と情報の重複が少ない

### 候補リスト

**#1. upset_level（波乱度Lv1-5）**
- Source: `netkeiba_upset_level.csv`
- 列: `upset_level` (1-5), `top_popularity_reliability` (0-100)
- AM8入手: ✓ 前日配信（shutuba.html）
- カバレッジ: 2020-2024（2025欠落 → scrape_missing_all.py で補填中）
- 期待効果: 荒れレース判定用、条件分類の補助
- 実装: race単位で全馬にbroadcast、2特徴量（upset_level_val, top_pop_reliability）

**#2. prev_review_score（前走不利短評スコア）**
- Source: `netkeiba_race_review.csv`
- 列: `review_score` （前走備考から自動スコア化、-3〜+3程度）
- AM8入手: ✓ 前走データなので完全オフライン
- カバレッジ: 2020-2025全年（277K行）
- 期待効果: 前走不利→巻き返し検出。v12.1で+0.00016（微小）だが2021年gap 0.0514で不採用。v15では情報量増でgap解消可能性
- 実装: 馬番×レース結合、horse_id経由で前走検索

**#3. training_eval_rank（調教評価ランクA-D）**
- Source: `netkeiba_training_eval.csv`
- 列: `training_rank` (A/B/C/D), `prev_review`, `training_intensity`, `training_move`
- AM8入手: ✓ 前日配信（newspaper.html）
- カバレッジ: 2024-2025（2020-2023は scrape_missing_all.py で補填予定）
- 期待効果: training_intensity_encとは別ソース（netkeiba独自評価）。既存調教特徴量と補完
- 実装: rank A=4, B=3, C=2, D=1 として数値化

**#4. master_index系（3分解指数）**
- Source: `netkeiba_master_index.csv`
- 列: `time_index` (総合), `start_index` (スタート), `chase_index` (追走), `agari_index` (上がり)
- AM8入手: ✗ 結果データなので**前走**に変換して使用（レース後確定）
- カバレッジ: 2025のみ → 全年取得中
- 期待効果: JRDB IDMと類似だが独立ソース。前走master_indexを前走特徴量として使用
- 実装: prev_time_index, prev_start_index, prev_chase_index, prev_agari_index（4特徴量）
- 注意: 現在カバレッジ低、全年取得後に採用検討

**#5. track_index系（netkeiba馬場指数）**
- Source: `netkeiba_track_bias.csv`
- 列: `track_index` (数値), `track_bias_text` (内/外/差しなど), `track_comment`
- AM8入手: ✗ レース後確定。**前走の馬場指数**として使用
- カバレッジ: 2020-2022+2025（2023-2024取得中）
- 期待効果: JRDB馬場差とは独立ソース。track_bias_textをone-hot化可能
- 実装: prev_track_index (数値) + prev_track_bias_inner/outer/front/back (4bool)

## 実装・検証プラン

1. **データカバレッジ確保**: scrape_missing_all.py 完了待ち（現在稼働中）
2. **特徴量追加実装**: `train/features_v15_new.py` と同形式で `train/features_v16_premium.py` 作成
3. **WF評価**: `train/train_v15_master.py` の ablation フレームで5候補を1個ずつ追加評価
4. **採用基準**:
   - WF mean AUC > 0.8858
   - 全年 AUC > 0.85
   - max (train-test) gap < 0.05
   - 5候補うちAUC改善に寄与する特徴量のみ採用
5. **採用後**: 150+α特徴量で v15.1 / v16 として本番デプロイ

## 本セッションでの進捗

- ✓ 未使用CSV 15種を棚卸し、モデル特徴量と対応付け
- ✓ 上位5候補を選定し、AM8取得可能性と実装方針を文書化
- ⨯ **実装・学習・検証は未実施**（4モデル WF 再学習で4-6時間必要）
- 次アクション: scrape_missing_all.py 完了後に data/netkeiba_master_index.csv / track_bias.csv 全年カバレッジ完成を待ち、ablation実行

## 注意事項

- 新特徴量追加は慎重に: v15 ベースラインが WF AUC 0.8856 と既に高水準
- `dam_top3r`・`prev_review_score`・`shinba_eval_score` は過去に採用基準未達で不採用（CLAUDE.md参照）
- 採用基準を満たさない場合、無理に追加すると過学習リスク
