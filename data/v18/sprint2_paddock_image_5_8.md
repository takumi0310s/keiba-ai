# Sprint 2 G: paddock_image_analyzer PoC (Session #47 G)

**作成**: 2026-05-08 (Session #47 G、 dev/sprint2)
**前提**: Session #42 E ultralytics + opencv-python install 済 (138 ms CPU 動作確認済)

---

## 1. 構成

`tools/paddock_image_analyzer.py` (180 行):
- YOLOv8n で 馬体検出 (CPU、 138 ms)
- bbox から 体格 / aspect ratio / pose score
- features 化 (5 件)

---

## 2. features (5 件)

```python
- paddock_horse_detected: 1/0
- paddock_max_conf: 検出 confidence
- paddock_body_size_score: bbox 面積 / 100000 (normalized)
- paddock_aspect_ratio: width/height (standing horse 1.3-2.5)
- paddock_pose_score: 1.0 if 標準 aspect、 0.5 else
```

---

## 3. 動作確認 (sample image)

```
target: zidane.jpg (人物画像、 馬なし)
inference_ms: 81.8
n_detections: 3 (person ×2、 tie ×1)
n_horses: 0  ← expected (馬なし画像)
features: all 0 (正常)
```

→ 環境動作確認 OK、 真のパドック画像で test 必要

---

## 4. production 統合 plan (Phase 4 7-8 月)

### 4.1 image source

- netkeiba パドック画像 (Premium 会員特典)
- JRA-VAN ネクスト 静止画
- (取得 plan は Session #42 E `docs/PHASE_4_VIDEO_FEASIBILITY_5_8.md` 参照)

### 4.2 features 拡張

```python
# Phase 4 で追加候補:
- DLC SuperAnimal で keypoint 推定
- 毛色 (健康状態 hint、 RGB 分析)
- 緊張度 (耳の位置)
- 馬体張り (筋肉 outline)
```

→ V21 (V20 + 動画) に 静止画 features も統合候補

---

## 5. caveat

- 真のパドック画像は 5/15 merge 後 取得
- ultralytics 8.4 + opencv-python は既 install (Session #42 E)
- torchvision NMS CUDA 不対応 → CPU 推論で対応 (138 ms 許容)

---

## 6. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only
✅ Phase 4 PoC は完全別 dir (data/v18/、 tools/)

→ **5/9 朝 V15 完全保証**

---

**Session #47 G 完了 (dev/sprint2)**
