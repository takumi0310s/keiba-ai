# Phase 16 C: パトロール YOLO PoC (8 features) (5/10)

> Session #87 Phase 16 C 領域
> 出力: tools/predict_core_v21.py の PATROL_YOLO_FEATURES (8 件)

---

## 1. 設計 model

| 項目 | 選定 |
|------|------|
| 物体検出 | ★ YOLOv8m / YOLOv11m ★ (馬 fine-tune) |
| 馬番 OCR | EasyOCR or Tesseract (post-bbox crop) |
| tracking | DeepSORT (馬群 occlusion 対応) |
| frame 抽出 | opencv 10 fps (1 R 60-120 秒 → 600-1200 frames) |
| GPU | NVIDIA RTX 4070 Ti SUPER 16GB ★ |
| Phase 提案 | YOLOv8m 採用、 v11 は 9/2 以降 検証 |

YOLOv8m 採用理由: pretrained COCO horse class (17) 動作確認済 (Session #42、 138ms CPU / GPU 高速)、 ultralytics 8.4.47 動作 OK。

---

## 2. 8 features 内訳

| # | feature | type | range | 検出 logic |
|---|---------|------|-------|----------|
| 1 | video_patrol_furi_count | int | 0-5 | 接触/進路カット/失速 集計 |
| 2 | video_patrol_route_position | float | 0-1 | 馬群最内 = 0、 最外 = 1 |
| 3 | video_patrol_loss_seconds | float | 0-3 | 失速時間 (理論 pace との差) |
| 4 | video_patrol_contact_severity | float | 0-1 | bbox 重なり frame 数 / 全 frame |
| 5 | video_patrol_block_count | int | 0-3 | 進路前 馬 occlusion 検出 |
| 6 | video_patrol_pace_loss | float | 秒 | 通過 timestamp loss |
| 7 | video_patrol_position_change | int | 順位変動 | tracking ID 順位 std |
| 8 | video_patrol_track_run_distance | float | m | 馬体軌跡 累積距離 |

---

## 3. inference pipeline (7/1+)

```
Step 1: video → opencv VideoCapture, 10 fps frame 抽出
Step 2: YOLOv8m (馬 fine-tune) で 全 frame 馬体 bbox 検出
Step 3: 馬番 OCR で bbox → umaban 紐付け (失敗時 IoU + tracking ID)
Step 4: DeepSORT で frame 間 tracking、 馬個体軌跡 構築
Step 5: 不利検出 logic:
  - 接触: bbox IoU > 0.3 連続 N frame
  - 進路カット: 軌跡 急変曲線 (角度差 > 30°)
  - 失速: 速度 std (10 frame window) 急増
  - 順位変動: tracking ID 順位 series
Step 6: 1 動画 → 18 馬 × 8 features
```

---

## 4. fine-tuning plan (5/15-9/2)

### 4.1 学習 schedule

| phase | 期間 | 内容 | sample size | GPU 時間 |
|-------|------|------|------------|---------|
| trial 蓄積 | 5/15-6/15 | パトロール動画 30-60 R | 30-60 | — |
| 馬体 annotation | 6/15-7/15 | bbox + 馬番 紐付け | 30-60 | — |
| YOLOv8m fine-tune | 7/15-8/1 | COCO horse → 馬画像 fine-tune | 60 R × 1000 frame | ★ 5-10h ★ |
| DeepSORT 統合 | 8/1-8/15 | tracking accuracy 検証 | 同上 | 3-5h |
| 不利検出 logic + validation | 8/15-9/2 | hold-out test | 同上 | 5-10h |
| **合計 GPU** | — | — | — | **13-25h** |

### 4.2 ハイパーパラメータ

```python
# YOLOv8m fine-tune
{
    'imgsz': 640,
    'batch': 16,
    'epochs': 100,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'data': 'horse_patrol.yaml',  # 馬体 + 馬番 class
}
```

### 4.3 期待精度

| metric | target | 現実的 baseline |
|--------|--------|----------------|
| 馬体 bbox mAP@0.5 | > 0.95 | 0.85 (zero-shot COCO horse) |
| 馬番 OCR acc | > 0.80 | 0.50 (Tesseract zero-shot) |
| tracking ID consistency | > 0.85 | 0.65 (DeepSORT pretrained) |
| 不利検出 F1 | > 0.6 | 0.35 (rule baseline) |

---

## 5. ★ 期待 corr 寄与 (V21) ★

| feature | 期待 corr | 用途 |
|---------|----------|------|
| furi_count | +0.005-0.010 | 前走補正 |
| contact_severity | +0.004-0.008 | 多頭数 R |
| block_count | +0.003-0.007 | 内枠不利検出 |
| pace_loss | +0.003-0.006 | 失速 → 次走補正 |
| ★ 平均寄与 ★ | ★ +0.020-0.040 ★ | — |

---

## 6. skeleton self-test

```
$ python tools/predict_core_v21.py
  C. パトロール YOLO (8): ['video_patrol_furi_count', 'video_patrol_route_position', ...]
```

→ ★ 8 features default fill OK ★ (動画なし、 不利なし default = 0 fill)

---

## 7. 環境 課題 (Session #42 確認済)

★ torchvision NMS CUDA 課題 ★:
- 現状: torchvision NMS が CUDA 非対応 → CPU fallback (138ms / frame)
- 7/1+ で `pip install torchvision --upgrade` で修正必要
- それまで CPU 推論で PoC (10 fps × 60 秒 = 600 frame × 138ms = 1.4 min/動画 許容)

---

## 8. V15 投資保護

✅ V15 production / model 不変
✅ predict_core_v21.py 新規
✅ 動画なし → default fill (不利なし) で予測継続

---

## 9. 結論

✅ C1: YOLOv8m + DeepSORT pipeline 確定
✅ C2: fine-tuning plan (30-60 R、 13-25h GPU、 7/15-9/2)
✅ C3: skeleton self-test pass (8 features default fill OK)
✅ C4: 期待寄与 +0.020-0.040 corr
✅ C5: torchvision NMS 課題 → 7/1+ 修正 plan
✅ C6: V15 完全保護
