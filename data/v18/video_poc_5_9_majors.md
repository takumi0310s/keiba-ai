# Session #47 E: 動画解析 PoC (任意、 5/8 13:00 以降)

## 1. 目的

5/9 (土) 重賞 (G1/G2/G3) のみ 調教動画 で 簡易 features 抽出。
今回 target: 京都新聞杯 (G2) 注目馬 3-5 頭。

PoC level (動作確認のみ)、 V20 / V21 で本格採用判定。

## 2. tool

`tools/video_analysis_poc.py`

```bash
python tools/video_analysis_poc.py --check-deps                      # deps check
python tools/video_analysis_poc.py --race-id 202608030611             # 京都 11R
python tools/video_analysis_poc.py --race-id 202608030611 --horse-ids 1 3 7  # 指定
```

## 3. 動画公開 timing (確認済)

- 5/9 のレース → 5/8 (金) **13:00** 公開
- G1: 全頭、 重賞: 注目馬のみ、 一般 R: 動画なし
- 公開後、 URL pattern が判明したら download 実装

## 4. 抽出 features (PoC)

| feature | 計算 | 期待意味 |
|---------|------|---------|
| frames_with_horse | YOLOv8 検出 frame 数 | 動画品質 sanity |
| body_size_score | bbox 面積 平均 | 馬体の充実度 proxy |
| pose_variability | bbox サイズ std / mean | 緊張度 proxy |
| stride_proxy | bbox 中心 x 速度 std | 歩様 proxy |

## 5. 必要 deps

- `ultralytics` (YOLOv8) — Session #41-42 で動作確認済 (138ms/frame)
- `opencv-python` (cv2)
- `torch` (CPU でも動作可、 GPU 推奨)

## 6. 動画 storage

- 出力先: `data/v18/videos/{race_id}_{horse_id}.mp4`
- size: 1 動画 ~ 30-50 MB
- 重賞 注目 5 頭 × 3 レース = ~750 MB
- **必ず .gitignore に追加** (push しない)

## 7. 5/8 13:00 以降 運用 step

```bash
# 13:00 動画公開確認
# 1. deps check
python tools/video_analysis_poc.py --check-deps

# 2. URL pattern 確認 (netkeiba スーパープレミアム会員 ログイン必須)

# 3. download_video() 関数を実装
#    - URL pattern 確定後
#    - Cookie 経由 download
#    - data/v18/videos/ に save

# 4. 京都新聞杯 注目馬 3 頭で PoC 実行
python tools/video_analysis_poc.py --race-id <race_id> --horse-ids <h1 h2 h3>

# 5. 出力 confirm
cat data/v18/video_poc_5_9_majors.json
```

## 8. 失敗時 (skip OK)

E は **学習目的のみ**、 失敗しても OK:
- 13:00 公開前 → skip
- URL pattern 不明 → skip
- ultralytics dep なし → skip
- 動画 download NG → skip

→ 5/9 V15 投資 / 5/10 verdict には **無関係**。

## 9. 採用判定 (Phase 4 に向けて)

- PoC 動作確認: 馬体検出 95%+ frame で成功
- 簡易 features 計算: ≥ 1 horse で features ≥ 1 個 取れる
- → V21 (9/1+) で本格採用判定

## 10. 関連 file

- `tools/video_analysis_poc.py` (本 PoC tool)
- `data/v18/videos/` (動画 storage、 .gitignore 対象)
- `data/v18/video_poc_5_9_majors.json` (出力)
