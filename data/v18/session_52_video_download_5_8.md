# Session #52 A: 動画 download 実装 (yt-dlp)

**作成**: 2026-05-08 23:XX (Session #52 A、 dev/training-poc)

## 1. 実装

`tools/video_downloader.py` (170 行):
- yt-dlp 2026.03.17 install OK
- Netscape cookies.txt 自動変換
- rate limit 3 秒/動画
- 失敗時 静止画 fallback (design)

## 2. 動作確認 (5/8 23:XX dry-run)

```
targets 3 (京都新聞杯 G2 / エプソムC G3 / 駿風 S OP)
cookies なし → Premium login 必要 (ユーザー manual)
全 dry_run、 yt-dlp install OK
```

## 3. 5/9 朝 ユーザー manual DL plan

```bash
# 1. Cookie refresh
python tools/refresh_cookie.py

# 2. video_downloader 実行
python tools/video_downloader.py --majors

# 3. data/v18/videos_5_9/ 配下に mp4 保存
```

## 4. ★ 投票方針 (絶対遵守) ★

5/9 重賞 投票なし、 動画 PoC は学習用。

## 5. V15 投資保護

✅ V15 model md5: 842b9a5f... 不変、 main 不変、 dev/training-poc 専用

→ **5/9 朝 V15 完全保証**

---

**Session #52 A 完了 (dev/training-poc)**
