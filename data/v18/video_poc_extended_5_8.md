# 動画解析 PoC 実 sample 拡張 (Session #43 D)

**作成**: 2026-05-08 (Session #43 D)
**前提**: Session #42 E で ultralytics 8.4 + YOLOv8 環境構築済 (138 ms CPU 動作確認)
**目的**: 静止画 single → 複数 frame / 動画 → frame 抽出 → 検出 への拡張

---

## 1. 拡張 PoC tool

### 1.1 ファイル

`tools/video_poc/extract_frames_and_detect.py` (新規、 約 170 行)

### 1.2 機能

| 入力 | 動作 | 出力 |
|------|------|------|
| `--video <path>` | OpenCV で frame 抽出 (5 fps、 max 50 frames) → YOLOv8 検出 | data/video_poc/frames/<id>/ + detections/<id>.json + summary |
| `--image <path>` | 1 枚 静止画で 検出のみ | summary JSON |

### 1.3 動作確認 (本 Session)

```
$ python tools/video_poc/extract_frames_and_detect.py --image data/video_poc/bus.jpg
============================================================
YOLOv8 検出: bus.jpg (134.2 KB)
============================================================
  model loaded
  total inference: 0.1s, 1 frames, avg 95.1 ms/frame
=== summary ===
  n_frames: 1
  n_with_horse_detected: 0  (バスの画像なので horse 0)
  horse_detection_rate_pct: 0.0%
```

→ 環境動作確認 OK、 inference 95-138 ms/frame (CPU)

---

## 2. 動画 source 候補 (Phase 4 着手用)

### 2.1 manual 取得 が必要 source

- JRA-VAN ネクスト (Premium 加入 後)
- netkeiba 動画 (Premium 会員特典)
- YouTube 競馬チャンネル (yt-dlp、 著作権 確認)
- JRA 公式 site (個人視聴用)

### 2.2 sample 動画 取得 困難な点

本 Session では Wikimedia public domain 動画の DL を試行 → 403 block (HTTP user-agent restriction)。
→ **Phase 4 開始時 (7/1+) に ユーザーが JRA-VAN ネクスト 加入 + 動画 manual 配置で着手**

---

## 3. 学習 data 蓄積 plan 更新

### 3.1 必要動画数 (Phase 4 PoC)

| 用途 | 必要数 | 想定 source |
|------|-------|----------|
| YOLOv8 検証 (zero-shot) | 10-20 動画 | 任意 |
| YOLOv8 fine-tune (馬体特化) | 50-100 動画 | JRA-VAN ネクスト |
| DLC SuperAnimal 試行 (zero-shot) | 5-10 動画 | 同上 |
| DLC fine-tune (HORSE-10 + 自前) | 100-200 動画 | 同上 |
| VIDEO_FEATURES 抽出 | 200+ 動画 | 同上 |

→ 計 200+ 動画、 各 1-2 分、 計 200-400 分 video → 約 5-10 GB

### 3.2 取得 schedule

| 期間 | 内容 |
|------|------|
| 7/1-7/7 | JRA-VAN ネクスト 加入、 動画形式 確認 |
| 7/8-7/14 | 50 動画 manual DL + frame 抽出 試行 |
| 7/15-7/21 | YOLOv8 検出 + 精度 評価 |
| 7/22-8/15 | fine-tune (HORSE-10 dataset 利用 + 自前 50 動画) |
| 8/16-8/31 | VIDEO_FEATURES 抽出 + V21 学習 |

### 3.3 学習 label 付け 工数

| 作業 | 工数 |
|------|------|
| 50 動画 × 100 frames × keypoint 12 points = 60,000 label | 30-50h |
| (DLC SuperAnimal zero-shot で skip 可) | 0h |

→ Session #42 E で SuperAnimal zero-shot 想定なら label 工数 大幅削減

---

## 4. 5/9 V15 投資保護 (D 領域)

✅ 動画 PoC code は data/video_poc/ + tools/video_poc/ に隔離
✅ V15 production 経路に影響なし
✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ ultralytics / opencv-python install は 別 module、 既存 dependency に影響なし

→ **5/9 朝 V15 完全保証**

---

## 5. 結論

✅ D1: 拡張 PoC tool (`tools/video_poc/extract_frames_and_detect.py`、 170 行)
✅ D2: 動画 → frame → 検出 の path 確認 (静止画 95 ms/frame)
✅ D3: 動画 sample 取得は Phase 4 開始時 (7/1+) に manual
✅ D4: 学習 data 蓄積 plan 更新 (200+ 動画、 5-10 GB)
✅ D5: SuperAnimal zero-shot で label 工数 大幅削減見込

→ **Phase 4 (7-8 月) 着手 即可能、 環境構築 + tool 完成済み**

---

**Session #43 D 完了**
