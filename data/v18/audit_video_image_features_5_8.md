# AUDIT-1 F: 動画 / 画像 features audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 F 領域)
**前提**: Session #43 D で YOLOv8 + ultralytics 8.4 環境動作確認済 (95-138 ms/frame CPU)
**位置付け**: read-only audit。 PoC は別作業

---

## 1. 取得可能な動画 / 画像 source

| source | URL / 入手 | timing | 加入要件 | 取得頻度 |
|--------|-----------|--------|---------|---------|
| netkeiba 調教動画 | premium 重賞 page | 金 13:00 | スーパープレミアム | 重賞のみ (週末 5-10 件) |
| netkeiba パドック動画 | premium レース page | 各 R 30 分前 | スーパープレミアム + マスター | 全 R |
| netkeiba パドック静止画 | レース page (誰でも) | 各 R 30 分前 | 不要 | 全 R |
| netkeiba レース動画 (過去) | レース result page | レース後 | スーパープレミアム | 全 R 全期間 |
| netkeiba 馬の写真 | db.netkeiba | いつでも | 不要 | 馬個体 |
| JRA-VAN ネクスト 動画 | (JRA-VAN 別契約 +1,000円/月) | レース前後 | JRA-VAN ネクスト | 全 R |
| YouTube JRA 公式 | youtube.com | レース後 | 不要 | レース後 |
| (法律 確認 必要) yt-dlp 競馬 ch | YouTube | 録画 後 | 不要 | 著作権 確認 |

---

## 2. 抽出可能 features (技術 / 期待効果)

### 2.1 YOLOv8 (馬体検出)

- **動作確認済** (Session #43 D): 95-138 ms/frame、 horse class 検出可
- **抽出**: bounding box、 confidence、 horse count
- **用途**:
  - 体格 score (馬の bbox 大きさ ratio)
  - 体型 (縦横比)
  - 馬数カウント (パドックで何頭)
- **期待 AUC**: +0.001-0.003 (体格 score 単独)

### 2.2 DLC SuperAnimal (姿勢推定 / keypoint)

- 馬 keypoint: 頭・首・肩・腰・各脚 (約 26 点)
- 学習: zero-shot で 動作 / fine-tune で精度向上 (DLC HORSE-10 ベース)
- **抽出**:
  - 歩様 (stride length、 stride freq、 cadence)
  - 姿勢 (head position、 neck angle、 back angle)
  - 左右非対称 (gait asymmetry)
  - 緊張度 (ear position、 head bobbing freq)
- **期待 AUC**: +0.005-0.010 (歩様 + 姿勢 5 features 合計)

### 2.3 OpenCV (色 / 体格 解析)

- HSV 抽出: 馬の毛色 解析 (汗ばみ / 艶 検出)
- contour: 体型 / 筋肉 plump 推定
- optical flow: 動き滑らか度
- **期待 AUC**: +0.001-0.003

### 2.4 動画固有 features

| feature | 抽出元 | 期待 |
|---------|--------|------|
| stride_length (歩幅 m) | DLC keypoint × 時間 | medium-high |
| stride_freq (歩数/秒) | DLC keypoint sequence | medium |
| gait_symmetry (左右非対称度) | DLC keypoint diff | medium |
| head_bobbing (頭振 frequency) | DLC head pos sequence | medium |
| ear_position (耳位置 score) | DLC ear keypoint | medium-low (緊張度) |
| posture_score (姿勢 score) | DLC neck/back angle | medium |
| coat_shine (艶) | OpenCV HSV | low |
| sweat_score (汗ばみ) | OpenCV HSV diff | low (パドック) |
| muscle_tone (筋肉 tone) | OpenCV contour | low-medium |
| nervousness_score (緊張度) | DLC + 動き解析 | medium |

---

## 3. 実装難度 + 期待効果 (matrix)

| 候補 | 実装難度 | 期待 AUC | 推奨時期 |
|------|---------|---------|---------|
| パドック静止画 (体格 score、 YOLOv8) | low (環境動作済) | +0.001-0.003 | Phase 4 PoC (7-8 月) |
| パドック動画 (歩様、 DLC) | medium (DLC fine-tune 必要) | +0.005-0.010 | Phase 4 (8-9 月) |
| 調教動画 (脚色、 重賞のみ) | medium-high | +0.005-0.010 (重賞限定) | Phase 4 後半 |
| レース動画 (corner 通過時 体勢) | high (動画 解析重い) | +0.001-0.003 | Phase 5 |
| 馬個体写真 (DB 蓄積) | low (静止画のみ) | low (一般情報のみ) | low priority |

---

## 4. 既存 PoC 状況

### 4.1 Session #43 D (5/8 早朝)

`tools/video_poc/extract_frames_and_detect.py`:
- OpenCV で frame 抽出 (5 fps、 max 50 frames)
- YOLOv8 で horse 検出
- 動作確認: bus.jpg (134 KB) で 95.1 ms/frame (CPU)

### 4.2 Session #44 (5/8 夜) 想定

PoC 拡張:
- 動画 sample 取得 課題 (Wikimedia 403 block)
- → Phase 4 開始時 (7/1+) に JRA-VAN ネクスト + 動画 manual 配置

### 4.3 学習 data 蓄積 plan

| Phase | 期間 | 動画 数 |
|-------|------|--------|
| PoC | 7/1-7/14 | 50 race × 30 動画 = 1,500 動画 |
| 学習 | 7/15-8/31 | + DLC fine-tune |
| V21 学習 | 9/1+ | V20 + 動画 features 5-7 件 |

---

## 5. リーク risk

| feature | timing | 用途 |
|---------|--------|------|
| パドック静止画 (30 分前) | live | Pattern B 候補 |
| パドック動画 (30 分前) | live | Pattern B 候補 |
| 調教動画 (金 13:00) | pre | Pattern A 候補 (前日まで) |
| レース動画 過去 | pre | 前走 features 候補 |

→ 全 features は post-race ではなく、 リーク risk なし

---

## 6. 期待効果 試算 (Phase 4)

| feature 件数 | 単独期待 | 累計期待 |
|------------|---------|---------|
| 1 (体格 score) | +0.001-0.003 | +0.001-0.003 |
| 5 (歩様 5 件) | +0.005-0.010 | +0.005-0.010 |
| 7 (歩様 + 姿勢 + 緊張度) | +0.005-0.012 | +0.005-0.012 |

V20 (6/8) WF AUC 0.890 + 動画 5-7 features → V21 (9/1) WF AUC **0.895-0.902** が 妥当な目標

---

## 7. 5/9 V15 投資保護

✅ video_poc は data/video_poc/ 別 dir、 V15 model 無関係
✅ 全 動画解析は Phase 4 (7/1+) で着手、 5/9-6/8 は完全 影響なし

---

## 8. 結論

✅ 動画 / 画像 全 source 一覧 + 抽出 features 候補 整理
✅ 期待 AUC: パドック静止画 PoC で +0.001-0.003、 動画 features 7 件で +0.005-0.012
✅ 投入時期: Phase 4 (7/1+ 学習 → 9/1 V21 投入候補)

**5/8-6/8 は無関係**、 Phase 4 開始時に 動画 sample 蓄積 を 着手
