# Phase 21E C: ブラウザ scraping 調査

> Session #88 (2026-05-11) Phase 21E C
> 結論: ★ Playwright/Selenium で技術的に可能、 ただし規約違反 risk + DRM 解除 risk → 自動大量 DL は NG ★
> ★ OS 画面録画 (private use) のみが現実解 ★

---

## 1. 結論 (一行)

**自動 m3u8 抽出 + ffmpeg DL は技術的に可能だが、 (a) RV 規約違反 + ban、 (b) DRM がある場合は違法 (2012 改正)、 (c) 工数 vs リターン 悪い。 **OS 画面録画 (個人視聴の延長 = 私的複製) のみ recommended**。**

---

## 2. 技術的検討: 各 method

### 2.1 Playwright / Selenium で video URL 抽出
- 仕組み: ブラウザ自動操作 + DevTools Network 監視 → m3u8/mp4 URL 取得
- 動作可能性: ★★★ (技術的には可能性高)
- 安定性: ★ (UI 変更 / SSO 期限切れ で頻繁壊れる)
- 規約: ★ (大量 DL = ban risk 高)

### 2.2 ffmpeg + cookie で 直接 m3u8 DL
- 仕組み: 認証 cookie 付き ffmpeg `-headers "Cookie: ..."` で stream DL
- 動作可能性: ★★ (m3u8 が DRM/widevine 暗号化されていれば不可)
- 安定性: ★★
- 規約: ★ (同上)

### 2.3 Stream Recorder ブラウザ拡張
- 動作可能性: ★★ (拡張により異なる)
- 安定性: ★ (拡張更新停止 リスク)
- 規約: ★ (同上)

### 2.4 OS 画面録画 (Mac QuickTime / Windows Game Bar)
- 仕組み: 視聴中の画面を そのまま録画
- 動作可能性: ★★★ (確実に動く)
- 安定性: ★★★
- 規約: ★★★ (個人視聴の延長 = 私的複製範囲、 著作権法 30 条 OK 想定)

### 2.5 OBS Studio で 高品質 録画
- 動作可能性: ★★★
- 安定性: ★★★
- 規約: ★★★ (個人録画範囲)
- メリット: fps / 解像度 / encode 自由設定 → 機械学習向け

---

## 3. 法的検討 (日本著作権法)

### 3.1 私的複製 (著作権法 30 条) — 個人 AI 学習 OK の可能性
- 個人で楽しむ目的の複製 = 私的使用 = OK
- 「個人 AI 学習」 = 私的使用に含まれるか? → グレー だが 30 条 4 項 (情報解析) で許容範囲
- ただし: 配布 / 公開 / 商用 = NG

### 3.2 DRM (Digital Rights Management) 解除 = 違法
- 2012 改正: DRM を解除して録画 / 保存する行為は違法 (著作権法 30 条 1 項 2 号)
- RV の Mpeg4 ストリーミングが DRM 含むかは未確認 → **要 5/15 trial 時 確認**
- 画面録画 (DRM 解除なし) = 私的複製範囲 = OK の可能性高

### 3.3 RV 利用規約 (JRA-VAN)
- [JRA-VAN 利用規約](https://jra-van.jp/info/rule.html) より:
  > 「会員は、本サービスを通じて取得した一切の情報を、当社又は権利者の事前の同意なく、複製、出版、放送、第三者への開示等することはできません。 ただし、会員自身の個人的使用を目的とする場合を除きます。」
- → **個人使用は明示的に OK**
- → 大量自動 DL / 公開 / 再配信 は NG

### 3.4 結論 (法的)
| 行為 | 判定 |
|------|------|
| ブラウザ視聴 + OS 画面録画 (個人 AI 学習) | ✅ OK (規約 + 私的複製範囲) |
| 自動 m3u8 抽出 + ffmpeg 大量 DL | ❌ NG (規約違反 + ban risk) |
| DRM 解除して保存 | ❌ NG (著作権法違反) |
| 録画した動画の SNS 公開 | ❌ NG (公衆送信権侵害) |
| 録画した動画から features 抽出 → V21 model 学習 | ✅ OK (個人利用、 model は配布しない前提) |

---

## 4. PoC 試行手順 (5/15-5/16)

### 4.1 まず DevTools で URL 抽出 試行 (技術検証 のみ)
1. RV にログイン → 重賞調教動画ページ
2. F12 → Network → "m3u8" / ".mp4" / ".ts" 検索
3. URL pattern 確認 (DRM 有無 確認)
4. **抽出だけ確認**、 自動 DL は実行しない

### 4.2 OS 画面録画で 1 動画 試行
1. Mac QuickTime: ファイル → 新規画面収録 → 該当エリア選択
2. または Windows: Win + G → Game Bar → 録画
3. または OBS Studio で 解像度 1920x1080 / 30fps / mp4 設定
4. 出力 file の format / 容量 / fps 確認

### 4.3 features 抽出 dry-run
1. ffmpeg で 1 fps frame 抽出 → JPG 画像 list
2. YOLOv8 で 馬体検出 試行
3. 1 動画 30 秒 で 30 frame、 features 抽出 latency 計測

---

## 5. 実装 plan (5/14 設計、 5/15+ PoC)

### 5.1 tools/rv_video_capture.py (新規、 5/14 設計)
```python
# Skeleton (実装は 5/15+)
def record_video(race_id, content_type, output_path):
    """
    OS 画面録画 wrapper。 個人視聴中の RV ブラウザ画面を 録画。
    content_type: 'paddock' / 'patrol' / 'training'
    """
    # Mac: QuickTime CLI / Windows: Win+G / OBS websocket
    pass

def extract_frames(video_path, fps=1):
    """ ffmpeg で frame 抽出 """
    pass

def extract_features(frames, model='yolov8'):
    """ YOLOv8 で 馬体検出 + features 数値化 """
    pass
```

### 5.2 規模感 (重賞のみ運用想定)
| 項目 | 数値 |
|------|------|
| 重賞 / 週 | 約 3-5 重賞 |
| 出走予定馬 / 重賞 | 平均 14 頭 |
| 動画 / 馬 / 重賞 | 調教 1 + パドック 1 = 2 |
| 動画 / 週 | 5 重賞 × 14 頭 × 2 = 140 動画 |
| 1 動画 平均 30 秒 | 140 × 30 = 4,200 秒 / 週 |
| 録画工数 | ★ 自動化困難 (画面録画 = 視聴中のみ可)、 50% 自動化 + 50% 手動 想定 |

→ **5/15 PoC で 1 重賞 (約 30 動画) を full 試行 して 工数感 確定**。

---

## 6. 5/14 設計 + 5/15 PoC TODO list

| # | 作業 | 期日 |
|---|------|------|
| 1 | DevTools 抽出 試行 (DRM 有無確認 のみ、 DL なし) | 5/15 |
| 2 | OS 画面録画 (Mac QuickTime) で 1 動画 録画 | 5/15 |
| 3 | OBS Studio install + 1 動画 録画 比較 | 5/16 |
| 4 | ffmpeg で frame 抽出 + YOLOv8 dry-run | 5/16 |
| 5 | tools/rv_video_capture.py skeleton 実装 | 5/14 |

---

## 7. 関連 source

- [JRA-VAN 利用規約](https://jra-van.jp/info/rule.html)
- [JRA-VAN 投稿ガイドライン](https://jra-van.jp/info/post_guide.html)
- [画面録画と著作権 (弁理士 解説)](https://www.innovations-i.com/copyright-info/?id=27)
