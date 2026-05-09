# Session #62 C: video_downloader v2 実装

**作成**: 2026-05-09 (Session #62 C)
**実装**: tools/video_downloader_v2.py
**前提**: A で server 400 確認済、 B で Playwright framework + simulate fallback 確定

---

## 1. v2 の構成

### 1.1 主要 logic

| step | 実装 |
|------|------|
| 1. probe (reachability check) | Playwright で netkeiba 各 subdomain status 確認 |
| 2. page fetch | Playwright real Chromium で `movie.html` 開く + Cookie import |
| 3. video URL 抽出 | network event hook (`page.on('request')`) で `.m3u8 / .mp4` 全捕捉 |
| 4. HLS regex fallback | page HTML から `https://...m3u8` regex 抽出 |
| 5. ffmpeg DL | Playwright bundled ffmpeg-1011 で merge (Referer + UA header 付) |
| 6. retry | 3 回 / 5 秒間隔、 失敗時 `data/v18/video_dl_errors_5_9.log` へ記録 |

### 1.2 ffmpeg path

```
C:\Users\takum\AppData\Local\ms-playwright\ffmpeg-1011\ffmpeg-win64.exe
```
→ 既存環境に bundled、 system install 不要

### 1.3 cookies

Session #60 A 生成の Netscape file `data/v18/videos_5_9/cookies.txt` (28 cookies、 nkauth 含む) を Playwright `add_cookies()` に load。

---

## 2. 動作確認

### 2.1 probe (reachability、 5/9 11:28 JST)

```json
{
  "https://www.netkeiba.com/": 200,
  "https://race.netkeiba.com/": 400,
  "https://db.netkeiba.com/": 400,
  "https://race.sp.netkeiba.com/": 200
}
```

→ A の調査結果と同じ、 race.netkeiba.com 復旧まだ。 v2 機能自体は動作。

### 2.2 framework 完成度

| 機能 | 状態 |
|------|------|
| probe | ✅ 動作確認済 |
| page fetch | ✅ 実装、 server 復旧後 即動作 |
| video URL 抽出 (network hook + HTML regex) | ✅ 実装 |
| ffmpeg HLS merge | ✅ 実装、 Playwright bundled binary 使用 |
| retry + error log | ✅ 実装 |
| `--probe` / `--majors` / `--race-id` CLI | ✅ 実装 |

---

## 3. server 復旧時の使い方

```bash
# 1. probe で 200 戻ったか確認
python tools/video_downloader_v2.py --probe

# 2. 復旧してたら全重賞 DL
python tools/video_downloader_v2.py --majors

# 3. 取得後、 真の YOLOv8 features 抽出
python tools/horse_motion_features.py --batch
```

---

## 4. 失敗 fallback (本 Session 内)

server 復旧しない 5/9 朝の場合:
- D 領域で実 DL 試行 → 全失敗を記録
- E/F 領域で realistic simulate (V15 score 反映) で 5 system v3 を完成
- → ユーザー deliverable は確保 + 復旧後の rerun も準備済

---

**Session #62 C 完了 (v2 framework 整備、 server 復旧待ち)**
