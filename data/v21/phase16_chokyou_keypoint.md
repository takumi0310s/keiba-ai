# Phase 16 D: 調教映像 keypoint detection PoC (10 features) (5/10)

> Session #87 Phase 16 D 領域
> 出力: tools/predict_core_v21.py の CHOKYOU_KEYPOINT_FEATURES (10 件)

---

## 1. 設計 model

| 項目 | 選定 |
|------|------|
| keypoint detection | ★ DeepLabCut SuperAnimal HORSE-10 ★ (zero-shot or fine-tune) |
| 代替 | MMPose (HRNet-W32 馬 fine-tune) |
| keypoint 数 | 17 (馬体: 鼻/耳/肩/肘/手根/球節/蹄/股/膝/飛節 等) |
| frame 抽出 | opencv 30 fps (調教 30-60 秒 → 900-1800 frames) |
| 動き解析 | numpy 軌跡 + scipy smoothing |
| GPU | NVIDIA RTX 4070 Ti SUPER 16GB ★ |

DLC SuperAnimal HORSE-10 採用理由: zero-shot で馬専用 keypoint 検出可、 fine-tune 工数 大幅削減 (Session #39 / #42 既確認)。

---

## 2. 10 features 内訳

| # | feature | type | range | 計算 logic |
|---|---------|------|-------|----------|
| 1 | video_chokyou_oikiri_time | float | 秒 | 開始 - 終了 timestamp |
| 2 | video_chokyou_last1f_speed | float | m/s | 終い 1F 通過 timestamp 差 |
| 3 | video_chokyou_smoothness | float | 0-1 | 蹄 keypoint 軌跡 jerk std (低 = smooth) |
| 4 | video_chokyou_form_score | float | 0-1 | 17 keypoint 全体配置 (理想形 cosine sim) |
| 5 | video_chokyou_stride_length | float | m | 同蹄 連続接地 距離 平均 |
| 6 | video_chokyou_gait_symmetry | float | 0-1 | 左右 step 時間差 std (低 = 対称) |
| 7 | video_chokyou_neck_drive | float | 0-1 | 鼻 keypoint 上下 振動 amplitude |
| 8 | video_chokyou_back_flex | float | 0-1 | 背中 keypoint (肩-腰) 角度変動 |
| 9 | video_chokyou_hindquarter_power | float | 0-1 | 飛節 推進角度 + 速度 |
| 10 | video_chokyou_finish_extension | float | 0-1 | 終い 1F の stride_length 増加率 |

---

## 3. inference pipeline (7/1+)

```
Step 1: video → opencv VideoCapture, 30 fps frame 抽出
Step 2: DLC SuperAnimal HORSE-10 で全 frame 馬 17 keypoint 検出
Step 3: keypoint 軌跡 構築 (時系列 numpy array、 17×2×N_frame)
Step 4: 軌跡 smoothing (savitzky_golay フィルタ)
Step 5: features 抽出:
  - oikiri_time: 開始 keypoint 動 frame〜終了
  - last1f_speed: 終い 1F = 200m 通過 timestamp 差
  - smoothness: 蹄 軌跡 jerk std (3 階差分 std)
  - form_score: 各 frame 17 keypoint 配置 vs 理想 template (cosine sim)
  - stride_length: 同蹄 接地 frame 検出 (y 座標 極小値) → 距離計算
  - gait_symmetry: 左右 step 時間差 std
  - neck_drive: 鼻 y 座標 amplitude
  - back_flex: 肩 - 腰 vector 角度 std
  - hindquarter_power: 飛節 角度 + 速度 (前進方向)
  - finish_extension: 後半 1F の stride 増加率
```

---

## 4. fine-tuning plan (5/15-9/2)

### 4.1 学習 schedule

| phase | 期間 | 内容 | sample size | GPU 時間 |
|-------|------|------|------------|---------|
| trial 蓄積 | 5/15-6/15 | 重賞調教動画 30-60 動画 | 30-60 | — |
| zero-shot inference | 6/15-7/1 | DLC SuperAnimal pretrained | 30-60 | 2-3h |
| fine-tune (必要時) | 7/1-7/15 | DLC HORSE-10 + 補助 annotation | 30-60 | 5-10h |
| 軌跡解析 logic + features | 7/15-8/15 | numpy / scipy 統合 | 30-60 | 3-5h (CPU) |
| validation | 8/15-9/2 | hold-out test | — | 3-5h |
| **合計 GPU** | — | — | — | **10-20h** |

### 4.2 fine-tune 必要性 判定 (6/15+)

zero-shot 精度 OK → fine-tune スキップ可:
- keypoint MAE < 5 px (224 input 換算)
- form_score corr_target > 0.05

不足時 → fine-tune 着手 (HORSE-10 dataset + 自前 annotation 30-60)

### 4.3 期待精度

| metric | target | zero-shot baseline (DLC) |
|--------|--------|--------------------------|
| keypoint MAE | < 5 px | 5-10 px |
| oikiri_time MAE | < 0.3s | 0.5s |
| stride_length MAE | < 0.3m | 0.5m |
| form_score corr_target | > 0.10 | 0.05 |

---

## 5. ★ 期待 corr 寄与 (V21) ★

| feature | 期待 corr | 用途 |
|---------|----------|------|
| oikiri_time | +0.005-0.012 | 既存 wood/sakaro time の補強 |
| last1f_speed | +0.005-0.010 | 終い speed corr 高 |
| smoothness | +0.003-0.007 | 全 R |
| stride_length | +0.005-0.010 | 距離適性 |
| gait_symmetry | +0.003-0.007 | 故障兆候 |
| その他 | +0.002-0.005 each | — |
| ★ 平均寄与 ★ | ★ +0.030-0.060 ★ | — |

---

## 6. skeleton self-test

```
$ python tools/predict_core_v21.py
  D. 調教 keypoint (10): ['video_chokyou_oikiri_time', 'video_chokyou_last1f_speed', ...]
```

→ ★ 10 features default fill OK ★

---

## 7. V15 投資保護

✅ V15 production / model 不変
✅ predict_core_v21.py 新規
✅ 動画なし → default fill で予測継続

---

## 8. 結論

✅ D1: DLC SuperAnimal HORSE-10 + numpy/scipy 軌跡解析 設計
✅ D2: 17 keypoint → 10 features 抽出 logic 確定
✅ D3: fine-tuning plan (30-60 動画、 10-20h GPU、 7/1-9/2)
✅ D4: skeleton self-test pass (10 features default fill OK)
✅ D5: 期待寄与 +0.030-0.060 corr
✅ D6: V15 完全保護
