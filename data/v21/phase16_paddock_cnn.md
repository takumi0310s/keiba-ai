# Phase 16 B: パドック CNN PoC (12 features) (5/10)

> Session #87 Phase 16 B 領域
> 出力: tools/predict_core_v21.py の PADDOCK_CNN_FEATURES (12 件)

---

## 1. 設計 model

| 項目 | 選定 |
|------|------|
| backbone | ★ EfficientNet-B3 (pretrained ImageNet) ★ |
| 代替 | ResNet50 / ConvNeXt-Tiny |
| head 数 | 12 (multi-head regression + classification) |
| input | 224×224 RGB frame |
| frame 抽出 | opencv 1 fps (動画 30 秒 → 30 frames) |
| frame 集約 | 平均 + 最大 (2 統計) |
| 学習 framework | PyTorch 2.11.0+cu126 |
| GPU | NVIDIA RTX 4070 Ti SUPER 16GB ★ |

EfficientNet-B3 採用理由: 224 input で良好な精度、 16GB GPU で batch_size=32-64 OK。 ResNet50 より省 memory。

---

## 2. 12 features 内訳

| # | feature | head type | range | 説明 |
|---|---------|-----------|-------|------|
| 1 | video_paddock_body_score | regression | 0-100 | 馬体充実度 (人手 annotation 平均) |
| 2 | video_paddock_sweat_level | classification (4 cls) | 0-3 | 発汗 level (無/微/多/異常) |
| 3 | video_paddock_tension_score | regression | 0-100 | 興奮度 (50 = 通常) |
| 4 | video_paddock_hoof_health | regression | 0-1 | 蹄健全度 |
| 5 | video_paddock_hindleg_drive | regression | 0-1 | 後肢踏み込み 強度 |
| 6 | video_paddock_calmness_score | regression | 0-1 | 落ち着き |
| 7 | video_paddock_coat_shine | regression | 0-1 | 毛艶 |
| 8 | video_paddock_ear_position | regression | 0-1 | 耳位置 stress (1=後) |
| 9 | video_paddock_head_carriage | regression | 0-1 | 頭の運び (1=理想) |
| 10 | video_paddock_back_arc | regression | 0-1 | 背中弓り (0=異常) |
| 11 | video_paddock_walk_rhythm | regression | 0-1 | 歩調 安定度 |
| 12 | video_paddock_overall_condition | regression | 0-1 | 総合 condition |

---

## 3. fine-tuning plan (5/15-9/2)

### 3.1 学習 schedule (現実的試算)

| phase | 期間 | 内容 | sample size | GPU 時間 |
|-------|------|------|------------|---------|
| trial 蓄積 | 5/15-6/15 | 個人録画 50-100 動画 | 50-100 | — |
| 人手 annotation | 6/15-7/1 | 12 head 別 annotation | 50-100 | — |
| pretraining | 7/1-7/15 | ImageNet → 馬画像 fine-tune (zero-shot 段階) | 100 | 5-10h |
| ★ multi-head fine-tune ★ | 7/15-8/15 | 12 head simultaneous fit | 100 | ★ 8-15h ★ (5070 epochs × 5 fold) |
| validation + tweak | 8/15-9/2 | hold-out test + hyper opt | 100 | 5-10h |
| **合計 GPU 時間** | — | — | — | **18-35h** |

### 3.2 ハイパーパラメータ (現状想定)

```python
{
    'optimizer': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'batch_size': 32,
    'epochs': 30,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 3,
    'augmentation': {
        'horizontal_flip': True,
        'random_crop': 224,
        'color_jitter': 0.2,
        'random_erasing': 0.1,
    },
    'loss': {
        'regression': 'huber (delta=0.1)',
        'classification': 'cross_entropy + label_smoothing 0.1',
    },
}
```

### 3.3 期待精度

| metric | target | 現実的 baseline (zero-shot) |
|--------|--------|----------------------------|
| body_score MAE | < 8 | ~15 |
| sweat_level acc | > 0.75 | ~0.5 |
| tension_score MAE | < 10 | ~20 |
| 全 head 平均 corr | > 0.5 | ~0.2 |

---

## 4. ★ 期待 corr 寄与 (V21) ★

| feature | 期待 corr | 寄与 大きい R |
|---------|----------|--------------|
| body_score | +0.005-0.010 | 全 R |
| sweat_level | +0.003-0.008 | 重賞 / 夏 |
| tension_score | +0.003-0.008 | G1 / 大舞台 |
| hindleg_drive | +0.003-0.007 | スプリント |
| ear_position | +0.002-0.005 | 全 R |
| その他 | +0.002-0.005 each | — |
| ★ 平均寄与 ★ | ★ +0.030-0.060 ★ | — |

---

## 5. skeleton self-test

```
$ python tools/predict_core_v21.py
[predict_core_v21] Phase 16 動画 features: 30 件
[predict_core_v21] V21 candidate: V20 (207) + 動画 (30) = 237 features
[predict_core_v21] models 利用可: {'paddock_cnn': False, 'patrol_yolo': False, 'chokyou_keypoint': False}
[predict_core_v21] OK: 全 30 features default 取得 成功
  B. パドック CNN (12): ['video_paddock_body_score', 'video_paddock_sweat_level', ...]
```

→ ★ 12 features default fill OK ★

---

## 6. 実 GPU 学習 status

★ 本 Phase 16 セッション内では実 GPU 学習を実行しない ★

理由:
- 実 sample 動画 = 0 (5/15+ trial 後に蓄積)
- 12-24h 学習時間 = 1 セッション内不可
- 人手 annotation 必要 (12 head × 50-100 動画)
- torchvision NMS CUDA 課題未解決 (Session #42 で確認、 7/1+ 修正)

→ 7/1-9/2 で 実学習 (18-35h GPU)、 9/2 V21 投入候補

---

## 7. V15 投資保護

✅ V15 production / model 不変
✅ predict_core_v21.py 新規 (V15 と完全独立)
✅ 動画なし → default fill で予測 可能 (V21 single fallback)

---

## 8. 結論

✅ B1: EfficientNet-B3 multi-head 設計 確定 (12 features)
✅ B2: fine-tuning plan (50-100 動画、 18-35h GPU、 7/15-8/15)
✅ B3: skeleton self-test pass (12 default fill OK)
✅ B4: 期待寄与 +0.030-0.060 corr
✅ B5: V15 完全保護
