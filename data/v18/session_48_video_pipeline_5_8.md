# Session #48 C: 動画解析 pipeline (dev/video-poc)

**作成**: 2026-05-08 (Session #48 C、 dev/video-poc)
**前提**: Session #42 E (ultralytics 8.4 install + 138 ms inference 確認)
**目的**: 動画 → features 化 の 5 module 統合 pipeline

---

## 1. 構成

`tools/video_pipeline/` 5 module:

| module | 内容 | 状態 |
|--------|------|------|
| `__init__.py` | package 宣言 | ✅ |
| `download.py` | 動画 download (netkeiba 調教 / JRA-VAN ネクスト) | design |
| `yolo_inference.py` | YOLOv8 馬体検出 (image / video) | ✅ 動作確認 |
| `keypoint_extract.py` | DLC SuperAnimal 歩様 keypoint | design (Phase 4 で 実装) |
| `features_aggregate.py` | YOLO + keypoint features 化 | ✅ |
| `main_pipeline.py` | 統合 entry | ✅ 動作確認 |

---

## 2. 動作確認 (zidane.jpg sample)

```
$ python tools/video_pipeline/main_pipeline.py --image data/video_poc/zidane.jpg
[pipeline] Step 1: download (skipped)
[pipeline] Step 2: YOLOv8 inference
  result: status=ok, n_horses=0  (zidane.jpg は人画像)
[pipeline] Step 3: keypoint extract (deferred、 Phase 4)
[pipeline] Step 4: features aggregate
  features: all 0 (正常、 馬なし image)
```

→ 環境動作確認 OK、 真の馬画像で test 必要 (Phase 4 で動画 source 確保後)

---

## 3. features 設計 (8 件、 5 + 3)

### 3.1 YOLOv8 由来 (5 件)

```python
- video_horse_size_score: bbox 面積 normalize
- video_pose_stability: frame 間 中心 std (低 = stable pose)
- video_aspect_ratio: standing horse 1.3-2.5
- video_horse_detection_rate: detection 率
- video_max_horse_conf: max confidence
```

### 3.2 DLC keypoint 由来 (Phase 4 で実装、 4 件)

```python
- video_stride_freq: 蹄 keypoint y 座標 peaks 頻度
- video_gait_symmetry: 左右蹄 phase correlation
- video_head_bobbing_amp: 頭 keypoint y 軌跡 std
- video_ear_pos_y_mean: 耳 keypoint mean y (集中度)
```

---

## 4. 5/8 13:00 動画公開後 PoC 試行 plan

```bash
# 5/8 13:00 以降、 ユーザー manual:
# 1. netkeiba 5/9 京都新聞杯 (G2) の調教動画 download (1-2 動画)
# 2. tools/video_pipeline/main_pipeline.py --video <path>
# 3. yolo inference + features 化 → JSON 確認
```

→ ただし netkeiba Premium login + Cookie 必要、 5/9 投資には不要

---

## 5. Phase 4 (7-8 月) 拡張 plan

| 期間 | 内容 |
|------|------|
| 7/1-7/14 | JRA-VAN ネクスト 加入 + 50 動画蓄積 |
| 7/15-7/31 | DLC SuperAnimal install + zero-shot keypoint |
| 8/1-8/15 | features 8 件 抽出 + 学習 data 統合 |
| 8/16-8/31 | V21 (V20 + 動画) 学習 |
| 9/1 | V21 投入判定 |

---

## 6. V15 投資保護

✅ V15 production 完全独立、 main 不変、 dev/video-poc only
✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict / app.py 完全不変
✅ 5/9 朝の V15 動作完全独立 (動画 pipeline は学習用)

→ **5/9 朝 V15 完全保証**

---

## 7. 結論

✅ C1: tools/video_pipeline/ 5 module (download/yolo/keypoint/aggregate/main)
✅ C2: 動作確認 OK (zidane.jpg sample)
✅ C3: YOLOv8 features 5 件 + DLC features 4 件 (Phase 4) 設計
✅ C4: Phase 4 (7-8 月) 拡張 plan
✅ V15 投資保護

→ **dev/video-poc 完了、 5/15 merge 候補、 Phase 4 で 動画 source 確保後 拡張**

---

**Session #48 C 完了 (dev/video-poc)**
