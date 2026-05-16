# Phase C: パトロール YOLO 5/18-5/24 PoC plan (詳細)

> Phase C (5/16) Terminal C 成果物
> 状態: ★ plan only、 actual 実行は 5/18 (日) から ★
> 前提 doc: [phase_c_patrol_8_features_spec.md](phase_c_patrol_8_features_spec.md)

---

## 1. PoC ゴール

★ 5/24 (土) までに **1 R 完全 pipeline (動画 → 18 馬 × 8 features csv)** を 動作させる ★

| 完成 条件 | 目標 |
|----------|------|
| 1 R 動画 取得 | OBS 録画 1 件 (60-120 秒) |
| frame 抽出 | ffmpeg 2 fps、 120-240 frame |
| YOLOv8 detection | 馬 bbox 検出 conf > 0.3 |
| ByteTrack tracking | 馬 ID 一貫性 > 50% |
| 8 features 計算 | 全 8 features 数値出力 (NaN なし) |
| csv 出力 | `data/v21/patrol_poc/<race_id>_patrol_features.csv` |
| 処理時間 | 1 R < 10 min (CPU)、 < 3 min (GPU) |

---

## 2. 日次 task table

### 2.1 5/18 (日) — 動画 source 取得 + 環境整備

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-10:00 | OBS Studio install + 設定 (1080p/30fps、 mp4 出力) | OBS 起動 OK、 test 録画 5 秒 mp4 取得 | 1h |
| 10:00-11:30 | JRA-VAN レーシングビュアー (RV) login + パトロール動画 navigate | パトロール動画 再生 確認 (5/17 開催 重賞 想定) | 1.5h |
| 11:30-12:30 | RV パトロール動画 1 R 録画 (OBS で 視聴中 録画) | mp4 file 1 件 (約 30-50 MB)、 60-120 秒 | 1h |
| 13:00-14:30 | 動画 file 配置 `data/v21/patrol_poc/<race_id>.mp4` | path 確定、 manifest.json 作成 | 1.5h |
| 14:30-16:00 | ffmpeg frame 抽出 PoC: 2 fps で frame_NNNN.jpg | 120-240 frame、 1920x1080 | 1.5h |
| 16:00-18:00 | ByteTrack install: `pip install supervision` (ByteTrack 含む) | import OK | 2h |

**5/18 完了物**:
- `data/v21/patrol_poc/<race_id>.mp4` (1 件)
- `data/v21/patrol_poc/<race_id>/frame_NNNN.jpg` (120-240 件)
- `data/v21/patrol_poc/manifest.json` (動画 metadata)
- ByteTrack 環境 OK

**工数 計**: 約 **8.5h**

---

### 2.2 5/19 (月) — YOLOv8 detection 動作確認

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-10:30 | YOLOv8s で 1 frame 検出 (zero-shot COCO horse class 17) | bbox 1-18 件 検出、 conf 出力 | 1.5h |
| 10:30-12:00 | 全 frame (120-240) で YOLOv8 推論、 結果 json 保存 | `<race_id>/detections.json` 作成 | 1.5h |
| 13:00-15:00 | bbox 可視化 (cv2 で 1 frame に bbox 描画) → 人 (騎手) と 馬 分離 確認 | 馬 = class 17 のみ filter、 騎手 (class 0) は別途 | 2h |
| 15:00-17:00 | 検出 数 安定性 確認 (frame 間 fluct 計測) | 平均 検出数 vs 出走頭数 比較、 < 20% fluct 期待 | 2h |
| 17:00-18:00 | 検出 課題 doc 化 (occlusion / 遠景 / 角度 等) | `data/v21/patrol_poc/detection_issues_5_19.md` | 1h |

**5/19 完了物**:
- `<race_id>/detections.json` (frame ごと bbox + conf)
- 可視化 sample 画像 (5-10 件)
- detection issues 報告 doc

**工数 計**: 約 **8h**

---

### 2.3 5/20 (火) — ByteTrack tracking 動作確認 (前半)

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-11:00 | supervision.ByteTrack を YOLOv8 出力に接続 | track_id 付き bbox 出力 | 2h |
| 11:00-12:00 | 全 frame で tracking、 結果 csv 保存 | `<race_id>/tracks.csv` (frame, track_id, x, y, w, h, conf) | 1h |
| 13:00-15:30 | track_id 一貫性 計測 (1 馬 = 1 track_id 持続率) | 馬数 × 一貫性 matrix | 2.5h |
| 15:30-17:30 | tracking 失敗 case 分析 (occlusion 多い / 急変区間) | 失敗 frame 一覧、 改善 idea | 2h |

**5/20 完了物**:
- `<race_id>/tracks.csv`
- tracking 一貫性 統計 (馬数 / 平均 持続率)

**工数 計**: 約 **7.5h**

---

### 2.4 5/21 (水) — ByteTrack tracking 動作確認 (後半) + 馬番紐付け 検討

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-11:00 | tracking パラメータ tune (track_buffer / min_consecutive 等) | 一貫性 50%+ 達成 | 2h |
| 11:00-12:00 | track_id → umaban 紐付け 案 検討 (OCR vs IoU vs manual) | 採用案 確定 doc | 1h |
| 13:00-15:00 | 馬番 OCR PoC: EasyOCR で bbox crop → 数字読み取り | 5-10 bbox で 試行、 認識率 計測 | 2h |
| 15:00-17:00 | OCR 失敗時の fallback: manual annotation tool 検討 (or IoU + jersey color) | fallback 案 確定 | 2h |

**5/21 完了物**:
- tracking tuned params
- 馬番 OCR 認識率 統計
- 紐付け fallback plan

**工数 計**: 約 **7h**

---

### 2.5 5/22 (木) — 8 features 計算 logic 実装 (前半 4 件)

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-10:30 | `compute_dangerous_pos_count` 実装 + unit test | track 1 件で 数値出力 | 1.5h |
| 10:30-12:00 | `compute_blocked_frame_count` 実装 + unit test | 同上 | 1.5h |
| 13:00-14:30 | `compute_track_change_count` 実装 + unit test | 同上 | 1.5h |
| 14:30-16:00 | `compute_jam_risk_score` 実装 + unit test | 同上 | 1.5h |
| 16:00-18:00 | 4 features を tracks.csv 全 track に適用、 結果 確認 | csv 出力 (馬数 × 4 col) | 2h |

**5/22 完了物**:
- `tools/v21/patrol_yolo_data_prep.py` の compute_* 関数 4 件
- 8 features csv (前半 4 col のみ)

**工数 計**: 約 **8h**

---

### 2.6 5/23 (金) — 8 features 計算 logic 実装 (後半 4 件)

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-10:30 | `compute_acceleration_late` 実装 + unit test | 数値出力 | 1.5h |
| 10:30-12:00 | `compute_position_recovery` 実装 (4 角推定 logic 込み) | 同上 | 1.5h |
| 13:00-14:30 | `compute_cornering_smoothness` 実装 | 同上 | 1.5h |
| 14:30-16:00 | `compute_kickback_dust_exposure` 実装 (色 segmentation) | ダート R で 数値出力、 芝 = 0 fill | 1.5h |
| 16:00-18:00 | 8 features full csv 出力 + 統計 sanity check | NaN なし、 range 内 | 2h |

**5/23 完了物**:
- `tools/v21/patrol_yolo_data_prep.py` の compute_* 関数 全 8 件
- 8 features csv (full、 馬数 × 8 col)

**工数 計**: 約 **8h**

---

### 2.7 5/24 (土) — 1 race 完全 pipeline + 検証

| 時間 | task | 完了 基準 | 工数 |
|------|------|---------|------|
| 09:00-11:00 | end-to-end pipeline 関数 統合 (`run_patrol_pipeline(video_path) -> csv`) | 1 R 単一 関数で実行 OK | 2h |
| 11:00-12:00 | 処理時間 計測 (CPU + GPU) | 1 R < 10 min (CPU)、 < 3 min (GPU) | 1h |
| 13:00-15:00 | 別 R (5/24 開催 重賞 想定) で 2 件目 録画 + pipeline 通し | 2 件 目 csv 出力 OK | 2h |
| 15:00-17:00 | 2 R features 統計比較 (range / std)、 sanity check | NaN なし、 features 分散 OK | 2h |
| 17:00-19:00 | PoC 結果 sum-up doc 作成 (`data/v21/phase_c_patrol_poc_result_5_24.md`) | 工数 / 精度 / 課題 / 次 step plan | 2h |

**5/24 完了物**:
- 完全 pipeline 関数
- 2 R 8 features csv
- PoC 結果 doc

**工数 計**: 約 **9h**

---

## 3. 工数 サマリー

| 日 | 工数 | 累計 |
|----|------|------|
| 5/18 (日) | 8.5h | 8.5h |
| 5/19 (月) | 8h | 16.5h |
| 5/20 (火) | 7.5h | 24h |
| 5/21 (水) | 7h | 31h |
| 5/22 (木) | 8h | 39h |
| 5/23 (金) | 8h | 47h |
| 5/24 (土) | 9h | **56h** |

★ **PoC 合計 約 56h** (1 週間、 1 日約 8h) ★

---

## 4. 出力物 一覧

### 4.1 ファイル

| path | 内容 |
|------|------|
| `data/v21/patrol_poc/<race_id_1>.mp4` | 5/18 録画 動画 1 R |
| `data/v21/patrol_poc/<race_id_2>.mp4` | 5/24 録画 動画 2 R 目 |
| `data/v21/patrol_poc/<race_id>/frame_NNNN.jpg` | 抽出 frame 群 |
| `data/v21/patrol_poc/<race_id>/detections.json` | YOLOv8 検出結果 |
| `data/v21/patrol_poc/<race_id>/tracks.csv` | ByteTrack 結果 |
| `data/v21/patrol_poc/<race_id>_patrol_features.csv` | 18 馬 × 8 features |
| `data/v21/patrol_poc/manifest.json` | PoC R metadata |
| `data/v21/patrol_poc/detection_issues_5_19.md` | detection 課題 |
| `data/v21/phase_c_patrol_poc_result_5_24.md` | PoC 結果 報告 |

### 4.2 script
| path | 内容 |
|------|------|
| `tools/v21/patrol_yolo_data_prep.py` | 5/16 skeleton (本 Phase C) → 5/18-5/24 で本実装 |

---

## 5. PoC 成功 判定

### 5.1 必達 (これを満たさなければ 5/25+ で 再 PoC)
- [ ] 8 features 全て NaN なし出力
- [ ] 1 R 処理時間 < 30 min (CPU)
- [ ] 2 R で 統計 sanity OK (features 分散あり、 全 0 ではない)

### 5.2 望ましい (達成なら V21 投入 path 加速)
- [ ] 1 R 処理時間 < 10 min (CPU)
- [ ] tracking 一貫性 > 50%
- [ ] YOLOv8 馬 検出 conf 平均 > 0.5
- [ ] 馬番 OCR 認識率 > 50%

### 5.3 stretch (達成なら 5/25+ 早期 fine-tune)
- [ ] 処理時間 < 3 min (GPU)
- [ ] tracking 一貫性 > 80%
- [ ] 馬番 OCR 認識率 > 80%

---

## 6. 投資保護 (絶対遵守)

| 項目 | 状態 |
|------|------|
| V15 .pkl.gz 不変 | ✅ |
| tools/predict_core.py 不変 | ✅ |
| tools/daily_predict.py 不変 | ✅ |
| app.py 不変 | ✅ |
| schtasks 既存 不変 | ✅ |
| 5/18-5/24 V15 production 継続 | ✅ (案 B 改 12R 1勝クラスのみ 2,100 円) |

---

## 7. risk + 撤退条件

| risk | level | mitigation / 撤退条件 |
|------|------|----------------------|
| OBS 録画 工数 想定超 (1 R > 30 分) | 中 | 5/18 工数計測、 1 R > 1h なら 5/22+ 半自動化 即実装 |
| YOLOv8 zero-shot 検出 精度 < 50% | 中 | HORSE-10 fine-tune 必要、 5/25+ にずらす |
| ByteTrack 一貫性 < 30% | 中 | DeepSORT に切替、 + 馬番 OCR 強化 |
| PoC 5/24 不完了 | 高 | 残課題は 5/25-5/31 に持ち越し、 V21 投入候補 9/1 は維持可 |
| 全体 PoC 失敗 (8 features 出力不能) | 低 | V21 投入 plan 全体 を 11/1 に1 ヶ月延期、 V20 単独 継続 (損なし) |

---

## 8. 5/25+ next step (PoC 後)

| 期間 | 内容 |
|------|------|
| 5/25-6/15 | 動画 source 蓄積 (30-60 R)、 半自動化 録画 script (OBS websocket) |
| 6/15-7/15 | 馬体 annotation (bbox + 馬番) 30-60 R |
| 7/15-8/15 | YOLOv8s fine-tune (HORSE-10 + 自前 annot)、 ByteTrack 統合 |
| 8/15-8/31 | 不利検出 logic validation (hold-out test)、 V21 学習 |
| 9/1 | V21 投入 判定 (WF AUC ≥ V20 + 0.005) |

---

## 9. 関連 doc

- [phase_c_patrol_8_features_spec.md](phase_c_patrol_8_features_spec.md) — 8 features 詳細仕様
- [yolov8_lfs_setup.md](yolov8_lfs_setup.md) — LFS 準備
- [phase21e_recommended_method.md](phase21e_recommended_method.md) — OBS 採用 + 5/14-18 PoC method
- [phase16_patrol_yolo.md](phase16_patrol_yolo.md) — 前身 設計 (Session #87)
- [phase16_summary.md](phase16_summary.md) — V21 candidate 237 features
