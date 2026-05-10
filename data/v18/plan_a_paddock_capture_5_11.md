# Plan A: netkeiba パドック動画 frame capture 実装 (5/11)

## 結論 (TL;DR)

`tools/paddock_video_capture.py` **完全動作確認** (5/11 01:18):
- 26 frame / 13 秒 / 1766 KB / errs=0
- 解像度 960x540、 JPEG q=85、 ~70 KB/frame
- 福島 12R 馬番16 ウインイザナミ (496kg-8) パドック映像 鮮明 capture

**netkeiba 規約 第 14 条 遵守設計**: 動画 file は保存しない、 ストリーミング再生中の iframe screenshot のみ抽出 (私的複製範囲)。

## 実装した tool

`tools/paddock_video_capture.py` (ASCII print, Windows cp932 OK)

```bash
# 動作モード
python tools/paddock_video_capture.py HORSE_ID --probe                  # DOM 調査のみ
python tools/paddock_video_capture.py HORSE_ID                          # 馬の paddock 履歴 index ページ
python tools/paddock_video_capture.py HORSE_ID --race-id RACE_ID        # 特定レース paddock viewer
python tools/paddock_video_capture.py HORSE_ID --race-id RACE_ID --fps 5 --duration 60
python tools/paddock_video_capture.py HORSE_ID --race-id RACE_ID --headless false
```

出力:
- `data/paddock_frames/{race_id}_{horse_id}/frame_NNNN.jpg` (3 fps × 30 秒 = 90 frame、 .gitignore 済)
- `data/paddock_frames/{race_id}_{horse_id}/manifest.json` (probe + frame timestamp)

## URL 構造 (実調査で判明)

| URL | 役割 |
|-----|------|
| `db.netkeiba.com/horse/paddock_movie.html?id=HORSE_ID` | 馬の過去 paddock 動画 **index** (一覧) |
| `db.netkeiba.com/horse/ajax_paddock_movie.html?id=HORSE_ID` | index の AJAX endpoint (JSON) |
| `race.netkeiba.com/race/paddock_movie.html?race_id=RACE_ID&id=HORSE_ID` | **viewer** (実際の動画 player) |
| `https://cdn.netkeiba.com/img/paddock/2026/{NNNNN}_0.jpg` | サムネイル静止画 |

認証: `nkauth` cookie + sp サブドメインで login session 必要。

## Dry-run 結果 (5/11 01:18、 完全動作)

ターゲット: ウインイザナミ (`2022106229`) × `202603010112` (4/11 福島12R)

### Step 1: cookie expire 検出 (00:30)
```
[PROBE] premium_gate=True
[AUTH] WARN: Premium_Regist_Box02 detected -> cookie expired
```

### Step 2: refresh_cookie.py 実行 (01:16)
```
NETKEIBA_EMAIL fallback 追加 (refresh_cookie.py)
+ data/cookies.json export 追加 (Playwright list 形式 69 cookies)
[OK] Cookie 更新成功 (28 重要 cookie 取得)
[OK] Premium 認証 OK
```

### Step 3: 完全動作確認 (01:18)
```
[PROBE] videoCount=0, iframes=9, premium_gate=False
[INFO] iframe video detected: iframe[src*="admint"]
[INFO] capture mode: iframe_screenshot
[OK] frames=26, size=1725.1 KB, errs=0
```

### 実 frame 内容 (frame_0010.jpg)
- 解像度 960x540
- 福島 12R / 馬番 16 / ウインイザナミ / 496kg (-8) / 石川 裕紀人 / 斤量 56
- パドック走行中の馬 + 引き手 + 観客 完全 capture

### 判明した m3u8 URL pattern
```
master:    race-player.netkeiba.com/5e4a7effafe26/media/{video_id}/{ts}.m3u8
540p variant: race-player.netkeiba.com/5e4a7effafe26/media/{video_id}/{ts}_540p_1555k.m3u8
```
540p / 1555 kbps、 admint.biz player 経由。 直接 m3u8 download も技術的には可能だが、 規約 grey なので screenshot 方式を維持。

## 次のステップ (production 化)

```bash
# 1. Cookie 期限管理 (auto)
python tools/refresh_cookie.py --auto    # 期限切れ時のみ refresh、 production OK

# 2. 当日 全レース 全頭 paddock 自動 capture (Phase 22 予定)
#    - daily_predictions の race_id × 出走馬 horse_id 全部回す
#    - 当日 18:00+ paddock 動画公開後に schtask で実行
python tools/paddock_video_capture.py HORSE_ID --race-id RACE_ID --fps 3 --duration 30

# 3. AI 特徴量抽出 (Phase 22+ 予定)
#    - YOLOv8 で馬 bbox 検出
#    - DLC SuperAnimal で姿勢推定
#    - gait / stride / posture 等を特徴量化
#    - 抽出 frame は破棄、 features のみ保存
```

## 規約 / 倫理 (重要)

netkeiba 利用規約 第 14 条:
> 第 14 条 私的利用範囲外の利用禁止

動画 file の **download は規約 grey 〜 NG**。
本 script は以下で対応:

1. **動画 file 保存しない**: ストリーミング再生中、 JS canvas drawImage で **frame のみ** 抽出
2. **frame は AI features 抽出用**: gait / posture / weight perception 算出 → 元 frame は破棄想定
3. **配布 NG**: `data/paddock_frames/` を `.gitignore` に追加 (本 commit)
4. **個人視聴のみ**: 共有 / 他人配布 完全禁止

これは「ストリーミング視聴 + 一時 cache」の私的複製範囲の解釈。
完全 clean を求めるなら **JRA-VAN ネクスト 加入** (月 ¥1,000、 7/1+ 予定 → 前倒し可)。

## frame 抽出 設計 (実装済)

```javascript
// CAPTURE_JS (Playwright page.evaluate)
const v = document.querySelector('video');
const c = document.createElement('canvas');
c.width = v.videoWidth;
c.height = v.videoHeight;
c.getContext('2d').drawImage(v, 0, 0);
return canvas.toDataURL('image/jpeg', 0.85);
```

Python 側:
- 1000/fps ms 間隔で `page.evaluate(CAPTURE_JS)` ループ
- `currentTime` 比較で同一 frame skip (重複防止)
- base64 → bytes → `frame_NNNN.jpg` 保存
- error 10 連続で abort

## 実測性能 (5/11 01:18)

| 項目 | 実測値 |
|------|--------|
| 1 馬 1 レース paddock 抽出 | 26 frame / 13 秒 / fps=2 |
| 解像度 | **960x540** (540p) |
| サイズ (jpeg q=85) | **~70 KB/frame** |
| 13 秒分 合計 | 1766 KB |
| 1 race (15 頭) 想定 | 15 × 1766 KB = ~26 MB (13秒分)、 30 秒 fps=3 で ~90 MB |
| 1 開催 24 races 想定 | ~2 GB (30 秒 fps=3) |
| 1 weekend 全 36 races 想定 | ~3 GB / 週 |

## DRM / 制限 リスク (実測で判明)

| 項目 | 結果 |
|------|------|
| DRM (Widevine) | **なし** (admint.biz HLS 平文 m3u8) |
| Aging restriction | アーカイブ動画は LIVE 30 秒制限なし、 13 秒抽出 100% 成功 |
| Cross-origin iframe | **canvas drawImage は CORS taint で詰まる**、 iframe.screenshot() が clean fallback |
| 同時セッション | 1 ブラウザ instance で逐次 OK、 並列は要検証 |
| 技術的には m3u8 直 download も可能 | だが規約 grey、 screenshot 方式を維持 |

## 完了範囲 (Phase 21D + 21E)

- ✅ tools/paddock_video_capture.py 実装 (270 行 → ~330 行に拡張)
  - canvas drawImage path (same-origin video 用)
  - **iframe screenshot fallback** (cross-origin 用、 canvas_taint 自動切替)
  - probe mode + capture mode
- ✅ tools/refresh_cookie.py 拡張
  - NETKEIBA_EMAIL fallback (.env 互換性向上)
  - data/cookies.json export (Playwright list 形式)
- ✅ .gitignore に data/paddock_frames/, data/cookies.json 追加
- ✅ data/v18/plan_a_paddock_capture_5_11.md (本 doc)
- ✅ **実 frame 抽出 dry-run 成功** (26 frame、 errs=0、 視覚確認 OK)

## V15 投資保護 (確認)

- predict_core.py / daily_predict.py / app.py / V15 model **完全不変**
- 新規 file のみ追加 (tools/paddock_video_capture.py、 plan_a doc)
- .gitignore 追記のみ
- 既存 V15 production pipeline 影響 0
