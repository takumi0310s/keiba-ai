# Phase 4 動画解析 AI PoC 設計 (Session #39 F)

**作成**: 2026-05-07 (Session #39 F)
**期間**: 2026-07月 〜 2026-08月 (V20 投入後)
**目的**: 調教動画 AI 解析で V20 を超える +5-15% ROI 改善を実現

---

## 1. 動機 + 仮説

### 1.1 既存 (V15-V20) のデータ限界

V15 〜 V20 は **構造化数値データ** (オッズ / 着順 / タイム / 血統 / 調教秒数) のみ:
- AUC 0.88-0.89 で plateau
- ROI 119.2% (V15 本番) → 140% (V20 想定) で頭打ち見込 ※ V15 ROI 119.2% は drift、 5/16 P0-1 真値 101.33% (docs/ROI_DISCREPANCY_2026_05_16.md)

→ **新次元のデータ source** が必要

### 1.2 動画解析の差別化価値

調教動画には人間トレーナー / プロ予想屋が **目視で読み取る情報** が含まれる:
- 馬の歩様 (smooth / jerky)
- 仕上がり (筋肉張り / 艶 / バランス)
- 加速性能 (加速時の脚捌き)
- 集中度 (耳の動き / 表情)

これら情報は構造化 features には含まれず、 V20 では未捕捉。

**仮説**: 動画 AI で抽出する features を V20 に追加で +5-15% AUC 改善見込

### 1.3 既存研究の根拠

馬獣医学 + AI 研究分野:
- 馬の跛行 (lameness) 検出: AI で精度 75-85% 報告例あり
- 馬の歩様分類 (gait classification): DeepLabCut + 自前 training で 80% 越え
- 動物姿勢推定の汎用 model 増加 (SuperAnimal-Quadruped 等)

→ 技術的に成立する見込

---

## 2. データ source

### 2.1 候補 source

| source | 提供 | 入手 channel | 量 | 質 |
|--------|------|------------|----|----|
| **JRA-VAN ネクスト** | 動画コーナー (有料) | 既存 加入 + 動画ダウンロード API | 全 JRA レース | 高 (公式) |
| **netkeiba 動画** | Premium 会員特典 | scraping (要 cookie) | 主要レース | 中-高 |
| **YouTube 一般** | 投稿者次第 | yt-dlp | 不安定 | 中 |
| **JRA 公式 site** | 当日のみ | scraping | 少 | 低-中 |

### 2.2 主軸: JRA-VAN ネクスト

- 既存 加入で利用可能 (Phase 3 で JV-Link 加入予定、 ネクスト追加で月額 +1,000円)
- 全 JRA レース動画 (発走 〜 ゴール、 1〜2 分/レース)
- 調教動画 (栗東 / 美浦) も提供
- スルー DRM/著作権 確認必須

### 2.3 調教動画 vs レース動画

**Phase 4 PoC は 調教動画 を主軸**:
- レース動画は "結果" 込み (post-race leak リスク)
- 調教動画 = pre-race の情報源、 学習・予測ともに利用可能

---

## 3. 技術スタック

### 3.1 動画 → 静止画 frame 抽出

```python
import cv2

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# 30 fps の動画から 5 fps (200 ms 間隔) でサンプリング
frames = []
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if i % int(fps / 5) == 0:
        frames.append(frame)
    i += 1
```

→ 1 分動画から 300 frames、 1 馬調教 (約 30s) → 150 frames

### 3.2 馬体検出 (Object Detection)

**第一候補: YOLOv8** (Ultralytics)
```bash
pip install ultralytics
```
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # nano、 高速
results = model(frame)
# class 17 = 'horse' (COCO)
horses = [r for r in results.boxes if r.cls == 17]
```

精度見込: 90%+ (COCO で horse class 学習済)

**第二候補: Detectron2** (Meta)
- より精緻な instance segmentation 可能
- GPU 推奨、 速度 YOLO < Detectron2
- Phase 4 後半で精度比較

### 3.3 姿勢推定 (Pose Estimation)

**第一候補: DeepLabCut** (動物特化)
```bash
pip install deeplabcut
```
- 動物姿勢推定の決定版 OSS
- 自前で keypoint label 付け → fine-tune
- 馬向け pretrained model 不在 → SuperAnimal-Quadruped で代替
- keypoint 例: 鼻 / 耳 / 肩 / 尻 / 4 脚 (前後 ×2) / 蹄 (4) = 計 12-18 points

**第二候補: SuperAnimal-Quadruped** (DLC + ImgT 連携)
- 4 足動物の汎用姿勢推定 model
- 馬を含む 39 種で pretrained
- 即時利用可、 fine-tune 不要

**第三候補: 自前 keypoint detection (Mediapipe + 馬用 fine-tune)**
- 工数大、 精度未知
- Phase 4 後半で検討

### 3.4 時系列特徴量抽出

frames から keypoints 時系列 → 特徴量化:

```python
# 各 frame の keypoints (12 points × 2 (x,y))
keypoints_seq = np.stack([extract_keypoints(f) for f in frames])  # (T, 12, 2)

# 特徴量 1: stride frequency (歩幅頻度)
# → 蹄 keypoint の y 座標が極小値を取る間隔
hoof_y = keypoints_seq[:, hoof_idx, 1]
peaks, _ = find_peaks(-hoof_y, distance=5)
stride_freq = 1.0 / np.mean(np.diff(peaks)) if len(peaks) > 1 else 0

# 特徴量 2: gait symmetry (歩様左右対称性)
left_hoof = keypoints_seq[:, hoof_left_idx, :]
right_hoof = keypoints_seq[:, hoof_right_idx, :]
gait_symmetry = compute_phase_correlation(left_hoof, right_hoof)

# 特徴量 3: head bobbing (頭振り → 跛行 indicator)
head_y = keypoints_seq[:, head_idx, 1]
head_bobbing = np.std(head_y - savgol_filter(head_y, 21, 3))

# 特徴量 4: ear position (耳の位置 → 集中度)
ear_avg_y = keypoints_seq[:, ear_idx, 1].mean()  # 低い = 後ろ向き = 集中減

# 特徴量 5: posture score (姿勢 score)
# → spine の curvature、 shoulder-tail line の傾き
posture_score = compute_posture_score(keypoints_seq)
```

### 3.5 model 統合

```python
# Phase 4 で抽出する features (Pattern A 相当、 pre-race のみ)
VIDEO_FEATURES = [
    'video_stride_freq',
    'video_gait_symmetry',
    'video_head_bobbing_amp',
    'video_ear_avg_y',
    'video_posture_score',
    'video_acceleration_rate',
    'video_muscle_definition_score',
    'video_coat_glossiness_score',
    'video_balance_score',
    'video_concentration_score',
]  # 計 10 features (Phase 4 PoC)
```

V21 model = V20 + VIDEO_FEATURES (10 features)

### 3.6 image classification model (フォールバック)

姿勢推定が困難な場合の代替:
- 全 frames を ResNet50 / ViT で encoding
- video-level pooling (mean / attention)
- 直接 binary classifier (top3 入着) を学習

→ **end-to-end 学習**、 解釈性低だが工数小

---

## 4. PoC 工数 + schedule

### 4.1 全体 schedule (7-8 月)

| 期間 | 内容 | 工数 |
|------|------|------|
| 7/1-7/14 | データ蓄積 (JRA-VAN ネクスト + netkeiba 動画) | 20-30h |
| 7/15-7/31 | YOLOv8 馬体検出 + DLC SuperAnimal 姿勢推定 動作確認 | 30-40h |
| 8/1-8/15 | 時系列特徴量抽出 + 自前 fine-tune (label 付け) | 40-60h |
| 8/16-8/31 | V21 学習 (V20 + VIDEO_FEATURES) + WF 検証 | 20-30h |
| 9/1+ | V21 投入 (or PoC NG なら V20 単独継続) | — |

合計: 100-200h、 業務外作業時間 (週末 10h × 8 週 = 80h で達成見込み)

### 4.2 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| データ蓄積完了 | 7/14 | 直近 1 か月の調教動画 50 レース分 (1,500 動画) |
| 姿勢推定動作確認 | 7/31 | 馬体検出精度 ≥ 80%、 keypoint 12 points × 80% 検出 |
| 特徴量抽出完了 | 8/15 | VIDEO_FEATURES 10 列、 全動画から欠損 < 30% |
| V21 学習完了 | 8/31 | WF AUC ≥ V20 + 0.005 (= 0.885+) |
| V21 投入判定 | 9/1 | LIVE retro winner_top1 ≥ V20 + 1pt |

---

## 5. リスク + 対策

| リスク | 確率 | impact | 対策 |
|--------|-----|--------|------|
| 動画品質 (解像度 / アングル) 不足 | 中 | 姿勢推定精度低下 | フィルタで高品質動画のみ使用、 4K 動画優先 |
| 著作権 / 利用規約 抵触 | 低-中 | 取得経路停止 | JRA-VAN ネクスト の利用規約確認 (個人 PoC 範囲) |
| GPU 不足 (Detectron2 / DLC fine-tune) | 中 | 学習工数 +数倍 | Google Colab / Kaggle Kernels で代替 (16GB GPU 無料) |
| keypoint label 付け 工数 | 高 | Phase 4 後ろ倒し | SuperAnimal-Quadruped で zero-shot、 label 付け回避 |
| 動画 features の improvement < +5bp | 中 | Phase 4 not GO | end-to-end ResNet50 で fallback、 解釈性犠牲で +改善見込 |
| 1 馬の動画が複数日に渡る | 高 | 集計方針未定 | 直近 N 日 (= 7-14 日) の平均で集計、 expanding window |
| V21 投入 NG | 中 | V20 単独継続 | V20 で +20% ROI 確保済、 V21 は upside 狙い |

---

## 6. 期待 ROI 改善

### 6.1 ベースライン (V20)

- WF AUC 0.885 (Phase 3 想定)
- 実 ROI 140% (戦略⑦込み、 Phase 3 想定)
- 月利 +5万〜10万円

### 6.2 V21 (V20 + 動画) 想定

| シナリオ | AUC | 実 ROI | 月利 |
|---------|-----|--------|------|
| 楽観 (姿勢推定良) | +0.010 (= 0.895) | 150% | +8万〜13万 |
| 中位 | +0.005 (= 0.890) | 145% | +6万〜11万 |
| 悲観 (動画 features 弱) | +0.000 (= 0.885) | 140% | V20 と同 |

→ 中位想定 でも +月利 1-3 万円改善、 PoC 工数 100-200h の元取り 6-12 ヶ月。

---

## 7. 関連 OSS + 参考資料

### 7.1 OSS

| 名 | URL | 用途 |
|----|-----|------|
| YOLOv8 | github.com/ultralytics/ultralytics | 馬体 bbox 検出 |
| Detectron2 | github.com/facebookresearch/detectron2 | instance segmentation |
| DeepLabCut | github.com/DeepLabCut/DeepLabCut | 動物姿勢推定 |
| SuperAnimal | github.com/DeepLabCut/DeepLabCut/tree/main/dlclibrary | 4 足動物 zero-shot |
| OpenCV | opencv.org | frame 抽出 / 前処理 |
| MediaPipe | google.github.io/mediapipe | 汎用姿勢推定 (馬向け fine-tune 必要) |

### 7.2 論文 / Web

- "Automatic detection of lameness in horses using deep learning" (Vetera, 2022)
- DeepLabCut animal pose estimation paper
- HORSE-10 dataset (将来検討)

詳細は `docs/PHASE_4_TECH_RESEARCH.md` (Session #39 G) 参照。

---

## 8. V15 / V20 動作不変保証

Phase 4 PoC は完全 sandbox:
- 新規 dir: `tools/video_ai/`, `data/video_ai/`
- V15 / V20 production 完全不変
- V21 投入時は V20 → V21 の段階的移行 (1 か月並行運用)

---

## 9. 結論

✅ Phase 4 動画解析 PoC 設計完了
✅ 技術スタック確定 (YOLOv8 + DLC SuperAnimal + 時系列 features)
✅ schedule (7-8 月、 工数 100-200h)
✅ milestone 5 段階定義
✅ リスク 7 項目 + 対策完備
✅ 期待 ROI 改善 (中位 +月利 1-3 万円)
✅ V15 / V20 動作不変保証

→ V20 投入 (7/1) 後、 7-8 月で Phase 4 PoC 着手。 9/1 V21 投入判定。

---

**Session #39 F 完了**
