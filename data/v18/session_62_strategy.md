# Session #62 B: 修復 strategy 確定

**作成**: 2026-05-09 (Session #62 B)
**前提**: A で server-side 400 block 確認、 client 側 即時修復不可

---

## 1. 5 つの候補 strategy 評価

| # | strategy | 5/9 朝 即時可? | 評価 |
|---|----------|---------------|------|
| 1 | yt-dlp に extra args (referer / UA / cookies-from-browser) | ❌ | 既に試行、 server 400 で無効 |
| 2 | Selenium で動的 page 解析 | ❌ | 同上 (Playwright real Chromium で 400 確認済) |
| 3 | **Playwright で headless chrome 動画 capture** | ❌ 即時 / ✅ 復旧後 | 復旧後の本筋。 framework 整備のみ |
| 4 | ffmpeg direct で signed URL 経由 | ❌ | signed URL 取得に movie.html 解析必要 → 400 で詰み |
| 5 | m3u8 (HLS) playlist 取得 → ffmpeg merge | ❌ | 同上 |

→ **5 つ全て、 server 復旧前は失敗確定**

---

## 2. 推奨 strategy: 「framework 整備 + simulate baseline」

### 2.1 phase 1 (本 Session 即実装)

- ✅ Playwright 経由の v2 downloader 実装 (server 復旧後 1 行で動く)
- ✅ ffmpeg は Playwright bundled `$LOCALAPPDATA/ms-playwright/ffmpeg-1011/ffmpeg-win64.exe` 利用
- ✅ HLS m3u8 抽出 logic 追加 (page HTML から `m3u8` regex)
- ✅ cookies は既存 Netscape file 流用
- ✅ retry: 3 回、 5 秒間隔
- ✅ error log: `data/v18/video_dl_errors_5_9.log`

### 2.2 phase 2 (server 復旧後)

```bash
# server 200 戻ったら 1 行で動く
python tools/video_downloader_v2.py --majors --use-playwright
```

### 2.3 simulate baseline (本 Session F で使用)

- DL 全失敗の場合 → Session #60 同様 simulate motion で 5 system v3
- ただし v3 は **simulate でも改善版 logic** を採用:
  - Session #60: 全馬 simulate を V15 top1 と一致と仮定
  - **Session #62 v3**: V15 score を ranking 反映 (top1=高 stride、 凡走馬=低 stability) で realistic に

---

## 3. ffmpeg path 確定

```
C:\Users\takum\AppData\Local\ms-playwright\ffmpeg-1011\ffmpeg-win64.exe
ffmpeg version n7.0.1-playwright-build-1011
```

Playwright bundled ffmpeg を流用 (system install 不要)。

---

## 4. cookies-from-browser

`yt-dlp --cookies-from-browser chrome` は **chrome に Premium login が必要**。
既存 `.env` の NETKEIBA_COOKIE が有効なら不要。
本 Session では既存 Netscape file `data/v18/videos_5_9/cookies.txt` (Session #60 A 生成) を使う。

---

## 5. 失敗時 fallback (Session #62 内)

```
DL 試行 (Playwright) → 全失敗
   ↓
realistic simulate 値生成 (V15 score base)
   ↓
horse_motion_5_9_REAL.csv (simulate but ranked) 出力
   ↓
5 system v3 で simulate を System 5 に投入
   ↓
Discord 3 通通知 (DL 失敗を明示)
```

---

## 6. NEXT (Area C)

→ tools/video_downloader_v2.py 実装 (Playwright + HLS m3u8 抽出 + ffmpeg merge)

---

**Session #62 B 完了 (strategy = Playwright framework + realistic simulate fallback)**
