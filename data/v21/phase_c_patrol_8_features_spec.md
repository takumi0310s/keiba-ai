# Phase C: パトロール 8 features 詳細設計 (5/18+ PoC)

> Phase C (5/16) Terminal C 成果物
> 目的: 不利検出 + 進路追跡 系の features を 8 件 設計
> 前身 doc: [phase16_patrol_yolo.md](phase16_patrol_yolo.md) (Session #87、 8 件)
> 本 doc: ★ 5/18+ PoC 用に refine、 user 提案 8 件 軸 + 既存 設計 統合 ★

---

## 1. 設計方針

### 1.1 入力
- パトロール動画 1 R (60-120 秒、 1920x1080、 OBS 録画 想定)
- 18 馬出走 (馬番 1-18) 想定
- 動画 fps: 元 30 fps、 features 抽出 fps: **2 fps** (= 1 R 約 120-240 frame)
  - 旧 phase16 案 (10 fps) から **2 fps に削減** = 計算量 80% 削減、 features 精度 影響少 (馬の動きは秒単位で十分捕捉可)

### 1.2 検出 pipeline
```
video.mp4
  ↓ ffmpeg (2 fps)
frame_0000.jpg, frame_0001.jpg, ... frame_NNNN.jpg
  ↓ YOLOv8s (馬 = COCO class 17、 zero-shot start)
bbox + conf list / frame
  ↓ ByteTrack or DeepSORT
horse_id 付き track (frame → horse_id → bbox)
  ↓ 馬番 OCR (option、 5/22+)
horse_id → umaban 紐付け (失敗時 IoU + manual annot)
  ↓ feature 計算 logic (本 doc Section 3)
patrol_8_features.csv (1 R × 18 馬 × 8 col)
```

### 1.3 fallback / null 処理
- 動画なし R → 全 8 features = `0` fill (= 不利なし扱い、 ★ 重要 ★)
- 馬番 OCR 失敗時 → tracking ID 順位で代替、 missing は `0` fill
- V21 model 投入時、 動画 features カバレッジ 30%+ 想定 (5/18+ PoC で確認)

---

## 2. 8 features 一覧 table

| # | feature 名 | type | range | 検出 logic 要約 | 既存案からの変更 |
|---|-----------|------|-------|----------------|----------------|
| 1 | `patrol_dangerous_pos_count` | int | 0-5 | 危険位置 (馬群最内 or 最外 + 直線含む) frame 数 | user 提案 ★ 新規 |
| 2 | `patrol_blocked_frame_count` | int | 0-N | 前方塞がれ frame 数 (前 5m bbox 占有) | user 提案、 旧 `block_count` 拡張 |
| 3 | `patrol_track_change_count` | int | 0-10 | 進路変更回数 (lateral 移動 量 閾値超え) | user 提案 ★ 新規 |
| 4 | `patrol_jam_risk_score` | float | 0-1 | 混戦 risk (周囲 5m 以内 馬 密度 平均) | user 提案 ★ 新規 |
| 5 | `patrol_acceleration_late` | float | pix/frame² | 終い加速度 (最終 25% 区間 速度差分) | user 提案、 旧 `pace_loss` 反転 |
| 6 | `patrol_position_recovery` | int | -17〜+17 | 4 角 → ゴール 位置回復 (順位差、 + が回復) | user 提案、 旧 `position_change` 拡張 |
| 7 | `patrol_cornering_smoothness` | float | 0-1 | 4 角 trajectory 平滑度 (角度差分 inv) | user 提案 ★ 新規 |
| 8 | `patrol_kickback_dust_exposure` | float | 0-1 | 砂被り time 比率 (前 2 m 内 砂 frame / 全) | user 提案 ★ 新規 |

### 2.1 既存 phase16 設計との差分

| 旧 (phase16_patrol_yolo.md) | 新 (本 doc Phase C) | 判断 |
|-------------------|--------------------|------|
| `video_patrol_furi_count` | (含意は `dangerous_pos_count` + `blocked_frame_count` に分散) | 統合 |
| `video_patrol_route_position` | (内包: `dangerous_pos_count` 危険位置 判定で代替) | 統合 |
| `video_patrol_loss_seconds` | `acceleration_late` (反転) | 改良 |
| `video_patrol_contact_severity` | `jam_risk_score` (5m 内 馬密度) | 改良 |
| `video_patrol_block_count` | `blocked_frame_count` (frame 単位 細分化) | 改良 |
| `video_patrol_pace_loss` | `acceleration_late` (反転) | 統合 |
| `video_patrol_position_change` | `position_recovery` (4 角 → ゴール 限定) | 改良 |
| `video_patrol_track_run_distance` | (削除、 features 飽和の懸念) | **削除** |
| (新規) | `track_change_count` | ★ 進路変更 単独 計測 |
| (新規) | `cornering_smoothness` | ★ 4 角 旋回 単独 計測 |
| (新規) | `kickback_dust_exposure` | ★ ダート 砂被り 単独 計測 |

→ 「不利検出 + 進路追跡」 軸で **より物理的 + 解釈可能な features** に refine。

---

## 3. 各 feature 詳細仕様

### 3.1 patrol_dangerous_pos_count
- **定義**: 馬群の **危険位置** (最内 ≤ 5% lateral or 最外 ≥ 95% lateral) に居る frame 数
- **計算**:
  1. 各 frame で 全馬 bbox 中心 x 座標 を 集計、 0-1 に normalize
  2. 対象馬の normalize 値が **≤ 0.05** or **≥ 0.95** なら count++
  3. 直線区間 のみ count (4 角は除外、 `cornering_smoothness` と分離)
- **range**: 0-5 (典型: 0-3 frame)
- **想定影響**: 内 (= ロス少 + 馬群閉じ込め risk)、 外 (= ロス大 + 開放) の 両極端 検出 → top3 確率に 影響
- **default fill**: 0

### 3.2 patrol_blocked_frame_count
- **定義**: 対象馬の **前方 5m 以内** に 他馬 bbox が 占有している frame 数
- **計算**:
  1. 対象馬 bbox 中心 を 起点に、 進行方向 (frame 間 diff) を 推定
  2. 進行方向 + 5m 以内 (pixel 換算 = 馬体長 × 2.5 程度) に 他馬 bbox center があれば count++
- **range**: 0-N (典型: 0-30 frame)
- **想定影響**: 進路カット / 詰まり 検出 → 次走 補正 (リカバリー期待)
- **default fill**: 0

### 3.3 patrol_track_change_count
- **定義**: lateral (横方向) 急変 回数
- **計算**:
  1. 対象馬 bbox center x 座標 の 5 frame 移動 mean diff を 計算
  2. abs(diff) > 閾値 (= 馬体幅 × 1.5、 約 50-80 pixel) なら 変更 count++
  3. 同方向 連続 は 1 count として 集約
- **range**: 0-10 (典型: 0-3)
- **想定影響**: 騎手の進路探し → 競馬上手 + 進路問題の両方を捕捉
- **default fill**: 0

### 3.4 patrol_jam_risk_score
- **定義**: 周囲 5m 以内の 他馬 密度 (平均)
- **計算**:
  1. 各 frame で 対象馬 bbox center から 5m radius 内に 他馬 bbox center があるか count
  2. 全 frame 平均 / max possible (= 17 馬) で normalize
- **range**: 0-1 (典型: 0.05-0.3)
- **想定影響**: 混戦 R で 不利 risk 上昇、 多頭数 R (16+) で 効く 想定
- **default fill**: 0

### 3.5 patrol_acceleration_late
- **定義**: 終い (動画 最終 25% 区間) の 加速度
- **計算**:
  1. 動画 全 frame の 75% 以降 を 「終い」 区間 とする
  2. 対象馬 bbox center x 座標 の 1 frame diff (速度) を 計算
  3. 終い区間 内で 速度 の さらなる diff (加速度) を 集計、 mean で 1 値化
  4. 単位: pixel / frame²、 正規化なし (絶対値で 騎乗評価)
- **range**: -10〜+10 pix/frame² 程度 想定
- **想定影響**: 「終い 伸びた」 = 上昇、 「失速」 = 下降 → top3 確率 に強影響 想定
- **default fill**: 0

### 3.6 patrol_position_recovery
- **定義**: 4 角通過時の 順位 → ゴール通過時の 順位 の **差** (+ が 回復、 - が 後退)
- **計算**:
  1. 4 角通過 frame を 推定 (動画 全体の約 60-70% time、 ★ 5/22+ で R 種別ごと calibrate ★)
  2. その frame で 全馬 順位 (横位置 + 進行方向 順) 算出
  3. ゴール通過 frame (動画 終端 5 frame) で 順位 算出
  4. recovery = 4角順位 - ゴール順位 (+ なら 上がった)
- **range**: -17 〜 +17 (典型: -3 〜 +3)
- **想定影響**: 「上がり 末脚」 「失速」 の 動画版 features (timing tracking より 直感的)
- **default fill**: 0

### 3.7 patrol_cornering_smoothness
- **定義**: 4 角 通過 trajectory の 平滑度 (= 1 - 急変度)
- **計算**:
  1. 4 角区間 (動画 50-70% time) の 対象馬 bbox center の 連続 frame を 抽出
  2. 各 frame 間の 角度差分 (heading) を 計算
  3. 角度差分 std を 正規化 (max = 90°)、 smoothness = 1 - std/90
- **range**: 0-1 (1 = 完璧 旋回、 0 = カクカク)
- **想定影響**: 騎手 + 馬の cornering ability → 距離 ロス と相関
- **default fill**: 0.5 (middle、 ★ 0 fill は smoothness 0 = 最悪 と解釈されてしまうため、 中央値 fill ★)

### 3.8 patrol_kickback_dust_exposure
- **定義**: 砂被り (前馬の砂が顔に当たる) の time 比率
- **計算**:
  1. **ダート R のみ** 有効 (芝は = 0 fill、 surface_enc で 判定)
  2. 対象馬の 前方 2m 以内 に 他馬 bbox + 「砂煙」 領域 (色 segmentation で 茶系 高頻度 領域) 検出
  3. 該当 frame 数 / 全 frame で比率算出
- **range**: 0-1 (典型: 0.0-0.3)
- **想定影響**: 砂被り 苦手馬 (= 通常 後方追走で 砂被って 失速) の 検出 → 次走 巻き返し期待
- **default fill**: 0
- **注**: 砂煙 segmentation は **PoC で精度確認必要**、 不可なら 「前 2m 内 他馬 frame 数」 で 簡易代替

---

## 4. 計算 logic implementation outline

### 4.1 必要 library
| library | 用途 | 5/16 時点 install 状態 |
|---------|------|----------------------|
| `ultralytics` | YOLOv8 推論 | ✅ 8.4.47 |
| `opencv-python` | frame 抽出、 色 segmentation | ✅ |
| `ffmpeg` | 動画 → frame 抽出 | ✅ (外部) |
| `numpy`, `pandas` | 数値計算 | ✅ |
| `bytetrack` or `deep-sort-realtime` | 馬個別 tracking | ❌ **要 install** (5/19+) |
| `easyocr` or `pytesseract` | 馬番 OCR (option) | ❌ 5/22+ |

### 4.2 pseudo code (詳細は skeleton script 参照)
```python
def compute_patrol_features(video_path: Path, race_id: str, surface_enc: int) -> pd.DataFrame:
    """1 動画 → 18 馬 × 8 features の DataFrame 返却."""
    frames = extract_frames(video_path, fps=2)
    detections = [detect_horses(f, model) for f in frames]  # YOLOv8
    tracks = track_horses(detections)  # ByteTrack
    # tracks: dict[horse_id, list[(frame_idx, bbox)]]

    rows = []
    for horse_id, track in tracks.items():
        feats = {}
        feats['patrol_dangerous_pos_count'] = compute_dangerous_pos(track, detections)
        feats['patrol_blocked_frame_count'] = compute_blocked(track, detections)
        feats['patrol_track_change_count'] = compute_track_change(track)
        feats['patrol_jam_risk_score'] = compute_jam_risk(track, detections)
        feats['patrol_acceleration_late'] = compute_acceleration_late(track)
        feats['patrol_position_recovery'] = compute_position_recovery(track, detections)
        feats['patrol_cornering_smoothness'] = compute_cornering_smoothness(track)
        feats['patrol_kickback_dust_exposure'] = compute_kickback(track, detections, surface_enc)
        feats['horse_id'] = horse_id
        feats['race_id'] = race_id
        rows.append(feats)
    return pd.DataFrame(rows)
```

---

## 5. 検証 plan (5/18+ PoC)

| step | 対象 | 検証項目 | 合格基準 |
|------|------|---------|---------|
| 5/18 | 1 R 動画取得 | OBS 録画 1 R (60-120 秒) | mp4 file 1 件 |
| 5/19 | frame 抽出 | ffmpeg 2 fps 抽出 | 120-240 frame |
| 5/20 | YOLOv8 detection | 馬 bbox 検出 | 1 frame で 8-18 bbox、 conf > 0.3 |
| 5/21 | tracking | ByteTrack で 馬 ID 一貫性 | ID 切り替わり < 30% |
| 5/22-5/23 | 8 features 計算 | 全 8 features 数値出力 | NaN / inf なし、 range 内 |
| 5/24 | 1 R full pipeline | 18 馬 × 8 features csv | 1 R 処理時間 < 5 min |

---

## 6. V21 model 投入時 投資保護

- ✅ V15 production / .pkl.gz / predict_core.py 完全不変
- ✅ 動画なし R → 8 features all 0 (or 0.5 for cornering_smoothness) fill → V15 と同等予測
- ✅ V21 model 学習時に 8 features が baseline AUC 0.8939 を下回るなら **採用見送り**
- 期待 corr 寄与 (★ 5/24 PoC 完成後 5/25+ で実測 ★): 平均 +0.020-0.040 (旧 phase16 想定維持)

---

## 7. risk + mitigation

| risk | mitigation |
|------|-----------|
| 動画 取得 工数 想定超 (1 R 30 分+) | ★ 5/18 工数計測、 1 R > 15 分なら 半自動化 即実装 ★ |
| YOLOv8 zero-shot で馬 detection 精度 不足 | HORSE-10 dataset で fine-tune (5/25+、 GPU 5-10h) |
| ByteTrack で ID 一貫性 < 50% | DeepSORT + 馬番 OCR で 補強 (5/22+) |
| 砂被り segmentation 精度 不足 | 簡易代替 (前 2m 内 他馬 frame 数) で 当面運用 |
| 動画 features の AUC 寄与 < +0.005 | V21 投入見送り、 V20 単独 継続 (絶対 損なし) |

---

## 8. 関連 doc

- [phase_c_patrol_yolo_poc_plan.md](phase_c_patrol_yolo_poc_plan.md) — 5/18-5/24 PoC plan 詳細
- [yolov8_lfs_setup.md](yolov8_lfs_setup.md) — LFS 準備
- [phase16_patrol_yolo.md](phase16_patrol_yolo.md) — 前身 (Session #87)
- [phase16_summary.md](phase16_summary.md) — Phase 16 全体 (237 features)
- [phase21e_recommended_method.md](phase21e_recommended_method.md) — OBS 録画 採用 method
