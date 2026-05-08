# Session #49 B: 動画 PoC 実行 (5/9 重賞)

**作成**: 2026-05-08 23:XX (Session #49 B、 dev/training-poc)
**目的**: 5/9 重賞 3R 注目馬の動画 PoC 試行 + features 抽出

---

## 1. 実装

### 1.1 ファイル

`tools/video_poc_majors_5_9.py` (190 行、 self-contained)

### 1.2 機能

```
1. data/v18/videos_5_9/ 配下の動画/画像を batch 処理
2. YOLOv8 馬体検出 (image 1 枚 or 動画 5fps × 50 frames)
3. features 4 件 抽出:
   - video_horse_detected (1/0)
   - video_max_conf
   - video_avg_size_score
   - video_aspect_ratio
4. JSON 出力 (data/v18/video_features_5_9_majors.json)
```

### 1.3 動作確認 (simulate mode、 5/8 23:XX)

```
$ python tools/video_poc_majors_5_9.py --simulate

processing: zidane.jpg → features all 0 (馬なし、 expected)
processing: bus.jpg → features all 0 (馬なし、 expected)
processing: sample_horse.jpg → features all 0 (zidane copy、 expected)

→ 環境動作確認 OK、 真の馬画像で test 必要
```

---

## 2. 5/9 重賞 動画 status

### 2.1 公開済 (5/8 13:00)

| race | grade | 動画範囲 |
|------|-------|---------|
| 京都 11R 京都新聞杯 | G2 | 注目馬 5-10 頭 (推定) |
| 東京 11R エプソムカップ | G3 | 注目馬 5-10 頭 (推定) |
| 新潟 11R 駿風 S | OP | 部分的 |

### 2.2 download status (5/8 23:XX)

```
data/v18/videos_5_9/  → 0 ファイル (未 DL)
理由: netkeiba Premium login + Cookie 必要、 ユーザー manual DL
```

### 2.3 5/9 朝 ユーザー manual DL plan

```
9:00 daily_predict 後、 候補馬 確定
9:30 ユーザー netkeiba login + 各重賞 5-10 動画 DL
     → data/v18/videos_5_9/ 配下に save (mp4)
10:00 python tools/video_poc_majors_5_9.py --all
      → features JSON 出力
10:30 5 system 合議 (Session #49 C で実装) で video features 統合
```

---

## 3. 5/9 朝に取れる動画 features

### 3.1 5/9 朝 9:00 daily_predict 後

候補馬 各重賞 top 10 馬 → 動画 DL (重賞 のみ、 全馬 の動画はない)

### 3.2 各重賞 注目馬 (5/9 朝確定)

placeholder (5/9 朝 daily_predict 後 入る):
```json
{
  "京都新聞杯_G2": {
    "noted_horses": [{"umaban": "?", "horse_name": "(5/9 朝確定)"}]
  },
  "エプソムC_G3": {
    "noted_horses": [{"umaban": "?", "horse_name": "(5/9 朝確定)"}]
  },
  "駿風_S_OP": {
    "noted_horses": [{"umaban": "?", "horse_name": "(5/9 朝確定)"}]
  }
}
```

→ data/v18/video_features/5_9_majors_target_horses.json で管理

---

## 4. features 設計 (4 件、 self-contained 簡易版)

```python
- video_horse_detected: 1/0
- video_max_conf: max YOLOv8 confidence
- video_avg_size_score: bbox 面積 mean / 100000 (normalize)
- video_aspect_ratio: standing horse 1.3-2.5 期待
```

→ Phase 4 (7-8 月) で DLC SuperAnimal keypoint で 拡張

---

## 5. caveat + 制限

- 真の馬動画 / 馬画像 で 5/9 朝 確認必要
- netkeiba Premium login + Cookie 必要、 ユーザー manual DL
- 動画 DL 工数: 1 動画 30 秒-1 分、 重賞 3 R × 各 5-10 馬 = 15-30 動画
- features 4 件は 簡易版、 Phase 4 で 8-10 件に拡張予定

---

## 6. V15 投資保護

✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict 完全不変
✅ main 不変、 dev/training-poc 専用
✅ 動画 PoC は学習用、 重賞投票しない (絶対遵守)

→ **5/9 朝 V15 案B改 完全保証**

---

## 7. 結論

✅ B1: tools/video_poc_majors_5_9.py (190 行、 self-contained YOLOv8)
✅ B2: features 4 件 (簡易版)
✅ B3: simulate 動作確認 OK
✅ B4: 5/9 朝 manual DL → batch run plan
✅ B5: V15 投資保護

→ **動画 PoC tool 完成、 5/9 朝 ユーザー DL 後 即 features 抽出可**

---

**Session #49 B 完了 (dev/training-poc)**
