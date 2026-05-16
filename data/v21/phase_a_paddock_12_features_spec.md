# Phase A — Paddock 12 Features Spec (V21)

Generated: 2026-05-16 (Session Phase A Day 1, Terminal A)
Source data: `data/video_ai_features/` (90 entries, 33 races, 85 horses)
Upstream artifacts: `body_condition_features.json` + `gait_features.json` + `yolov8_features.json`

## 設計方針

V15 (150 features) を 不変 とし、 paddock 動画から導出可能な horse-level な 12 features を新規追加する。

既存 video_ai_features の生成元:
- `yolov8_features.json` — bbox per frame (n_horses, conf, w, h, cx, cy, area)
- `gait_features.json` — bbox 時系列の集約 (aspect mean/std/range, motion speed, area change)
- `body_condition_features.json` — frame-level の coat/body/color/condition_score の集約 (aggregated + per_frame)

これら 3 系統 から 12 features を 派生 する。 直接 keypoint (DLC など) は未取得のため、 keypoint 系 (head_bobbing, ear_position, tail_movement, hoof_lift) は **gait の bbox 動き から proxy** とする。

## 12 Features 一覧

| # | Feature 名 | derivation source | 計算式 | 直感 |
|---|---|---|---|---|
| 1 | `pad_body_condition_score` | body_condition.aggregated.condition_score_mean | float | 馬体の全体スコア (0-1) |
| 2 | `pad_body_condition_std` | body_condition.aggregated.condition_score_std | float | スコア揺らぎ (健全性 vs 興奮) |
| 3 | `pad_coat_gloss` | body_condition.aggregated.coat_brightness_mean | float | 毛艶 (明度) |
| 4 | `pad_coat_contrast` | body_condition.aggregated.coat_contrast_mean | float | 毛色コントラスト (筋肉陰影) |
| 5 | `pad_body_aspect_mean` | body_condition.aggregated.body_aspect_mean | float | 体型 (横長 / 縦長) 平均 |
| 6 | `pad_body_aspect_var` | body_condition.aggregated.body_aspect_std | float | 体勢変化幅 (歩行 vs 停止) |
| 7 | `pad_body_compactness` | body_condition.aggregated.body_compactness_mean | float | 体の引き締まり度 |
| 8 | `pad_gait_motion_speed` | gait.features.motion_speed_mean | float | 重心移動速度 (歩幅 proxy) |
| 9 | `pad_gait_motion_std` | gait.features.motion_speed_std | float | 動きの不安定さ (興奮 proxy) |
| 10 | `pad_gait_aspect_range` | gait.features.aspect_range | float | 体勢変動 range (歩様 proxy) |
| 11 | `pad_gait_area_change` | gait.features.area_change_mean | float | 体表面積変化 (息遣い / 体勢) |
| 12 | `pad_yolo_conf_mean` | gait.features.conf_mean (or per_frame agg from yolov8) | float | 馬体 detection 信頼度 (映りの質) |

## 欠損戦略

- 90 entries に存在しない (race_id, horse_id) 組: 全 12 features NaN
- LGB は NaN を自動 split 学習可、 XGB は `missing=np.nan` 指定で対応
- 補完しない (mean fill すると定数列化、 signal 死亡)

## 既知の制約

1. **coverage 極小**: 90 entries / V15 cache 527,280 行 = **0.017%** (race_id+horse_id 完全一致)
2. **video は 2026 のみ**: V15 cache は 2015-2025 のため、 (race_id, horse_id) 完全一致では **0 overlap**
3. **horse_id 共通分**: 85 video horses 中 72 が cache (2015-2025) に過去出走履歴あり (397 rows)、 ただし 過去 race に未来 paddock features を当てるのは leak、 PoC では NG
4. **keypoint 未取得**: head_bobbing / ear_position / tail_movement / hoof_lift は DLC 等未導入のため proxy しかない (gait の bbox 動きで近似)

## 5/31 までの coverage 拡大 plan

V15 cache 2025 年 fold (~100K rows) でも有意な signal を得るためには **少なくとも 1-2% coverage** (= 1,000-2,000 races の 動画 features) が必要。

- 現状: 33 races
- 目標: 1,000+ races by 5/31 (15 days)
- ペース: **約 65 races/day の 動画取得** + features 抽出

= 1日 net で 130-180 動画 (paddock + patrol) を JRA RV / netkeiba 動画 から 取得し yolov8/gait/body_condition pipeline を流す必要あり。

5/31 後の Phase B 学習: V15 cache に 2025 年末 ~ 2026 年初 の 1,000-2,000 races の paddock features を merge し、 fold 25 (2025 末) で評価。

## 出力 CSV (paddock_12_features.csv) schema

```
race_id, horse_id, pad_body_condition_score, pad_body_condition_std,
pad_coat_gloss, pad_coat_contrast, pad_body_aspect_mean,
pad_body_aspect_var, pad_body_compactness, pad_gait_motion_speed,
pad_gait_motion_std, pad_gait_aspect_range, pad_gait_area_change,
pad_yolo_conf_mean
```

- `race_id`: 12-digit netkeiba 形式 (例: `202603010112`)
- `horse_id`: 10-digit netkeiba 形式 (例: `2022106229`)
- 12 float columns、 NaN は欠損のまま

## key file 一覧

- 仕様: `data/v21/phase_a_paddock_12_features_spec.md` (本 file)
- merger: `tools/v21/paddock_features_merger.py` (新規)
- 出力 CSV: `data/v21/paddock_12_features.csv` (90 行想定)
- PoC trainer: `tools/v21/train_v21_paddock_poc.py` (新規)
- 結果 JSON: `data/v21/phase_a_poc_result.json`
- 結果 MD: `data/v21/phase_a_poc_result.md`

## V15 (production) との分離

絶対に変更しない (再掲):
- `keiba_model_v135_*.pkl.gz` 系の V15 production weights
- `tools/predict_core.py` / `tools/daily_predict.py` / `app.py`
- `train/features_v15_new.py` / `train/train_v15_master.py`

本 Phase A は 全て新規 file を `tools/v21/` / `models/v21/` / `data/v21/` 配下にのみ作成する。
