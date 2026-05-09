# 動画 features 抽出 logic 設計 (V21 用)

> Session #80 (2026-05-09) 作成
> source: JRA-VAN RV (主) + BS 録画 (副) + パドック静止画 (補助)

## 全体 pipeline

```
動画 (mp4 / mov、 30 秒〜2 分)
    ↓
frame 抽出 (30 fps → 1 fps、 30-120 frames/動画)
    ↓
YOLOv8 馬体 detection (bbox + class confidence)
    ↓
frame 単位 features 計算 (5 features)
    ↓
動画単位 集計 (mean / std / min / max)
    ↓
統合 score (重み付き sum) + raw 5 features → V20 ensemble に組み込み
```

## detection 仕様

| 項目 | 値 |
|------|----|
| Model | YOLOv8m (8.4) |
| pre-trained | COCO horse class (id=17) |
| fine-tune | 不要 (zero-shot で OK、 必要なら HORSE-10) |
| input size | 640x640 |
| confidence threshold | 0.5 |
| inference time | 95-138ms / frame (GPU) |
| frame サンプリング | 30fps → 1fps (30 倍 sub-sample) |

## 特徴量 5 個

| # | 特徴量 | 算出方法 | 範囲 | 解釈 |
|---|--------|---------|------|------|
| 1 | `stride_length` | bbox 中心 X 座標の 連続 frame 間 移動量 (mean) | 0-100 | 大: 歩幅広い (好調 sign) |
| 2 | `body_size_relative` | bbox 面積 / frame 全体面積 (mean) | 0-1 | 大: カメラ近接 (張り良い 馬体) |
| 3 | `stability_score` | bbox 中心 Y 座標の 連続 frame 間 std (低いほど安定) | 0-50 | 低: 安定歩様、 高: 不安定 |
| 4 | `tension_score` | confidence の std + frame 間 bbox 急変動 frequency | 0-1 | 低: 落ち着き、 高: 緊張・興奮 |
| 5 | `pace_score` | stride_length の std (rhythm 安定度) | 0-50 | 低: rhythm 一定 (好調)、 高: rhythm 乱れ |

## 動画単位 集計

各 raw features に対し:
- `mean` (主、 corr 期待大)
- `std` (rhythm 評価)
- `min` / `max` (極端値検出)

→ 5 features × 4 集計 = 20 raw features

## 統合 score (重み付き sum)

```python
video_score = (
    0.30 * stride_length_mean   # 歩幅広い
  + 0.20 * body_size_mean        # 体格張り良い
  - 0.20 * stability_score_mean  # 安定 (符号反転)
  - 0.15 * tension_score_mean    # 落ち着き (符号反転)
  - 0.15 * pace_score_mean       # rhythm 一定 (符号反転)
)
```

→ 1 features (`video_score`) + raw 20 features = 21 features 投入

## V20 ensemble 組み込み 設計

```
V20 (LGB + XGB + FT + IR、 4-model ensemble)
    ↓
動画 features 21 個 を 全 4 model に投入
    ↓
V21 (V20 + video features、 同 4-model ensemble)
    ↓
重賞 R: V21 を使用 (動画あり)
一般 R: V20 single (動画なし、 fallback)
```

## 学習データ構築

| source | 期間 | 想定枚数 |
|--------|------|---------|
| RV (重賞調教動画) | 2025-05-15 以降 trial 中 | 30-50 R × 16-18 馬 = 約 500-900 frame セット |
| BS 録画 (グリーンチャンネル) | 2025-05-15 以降 録画 | 100-200 R × 同等 = 約 1,500-3,000 frame セット |

→ V21 PoC: trial 1 ヶ月で 2,000-4,000 frame セット 集約 → 学習可能

## カバレッジ問題 と 補完 strategy

| R 区分 | 動画あり率 | V21 適用 |
|--------|-----------|---------|
| G1/G2 | 100% (RV + BS) | 全 R で V21 使用 |
| G3 | 100% (RV + BS) | 全 R で V21 使用 |
| OPEN/L 特別 | 30-50% (BS のみ) | 動画あり R のみ V21、 ない R は V20 |
| 一般 R | < 10% (一部 BS) | 基本 V20、 動画あり R のみ V21 |

→ ハイブリッド運用: V20 (default) + V21 (動画あり R のみ)

## 想定 効果

| 指標 | V20 | V21 (動画あり R) | delta |
|------|-----|----------------|-------|
| AUC | 0.90025 | 0.92-0.93 | +0.020-0.030 |
| winner_top1 | 36-38% | 38-41% | +2-3pt |
| ROI (戦略⑦込み) | 145-150% | 155-165% | +10-15pt |

## risk

| risk | 対策 |
|------|------|
| RV 動画品質 不足 (cropped / 角度不一致) | YOLOv8 confidence threshold 調整、 複数 frame 平均化 |
| frame 抽出 重い (1 動画 30 秒) | 1fps sub-sample で 計算量 1/30 |
| 学習 data 少 (< 1,000 sample) | trial 期間延長、 BS 録画 で補完 |
| 規約 違反 | 公開禁止、 個人 AI 学習限定、 自動 DL しない |

## 関連 doc
- [JRA_VAN_RV_TRIAL_GUIDE.md](JRA_VAN_RV_TRIAL_GUIDE.md) — RV trial 手順
- [PHASE_4_VIDEO_REPLAN_v2.md](PHASE_4_VIDEO_REPLAN_v2.md) — Phase 4 plan v2
- [RV_TRIAL_5_15_CHECKLIST.md](RV_TRIAL_5_15_CHECKLIST.md) — 5/15 trial 開始 checklist
- [PHASE_4_VIDEO_FEASIBILITY_5_8.md](PHASE_4_VIDEO_FEASIBILITY_5_8.md) — feasibility 検証 (Session #42)
