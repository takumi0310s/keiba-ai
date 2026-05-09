# Session #60 A: Cookie 状態確認 + .env → JSON 変換

**作成**: 2026-05-09 (Session #60 A、 5/9 朝 9:30+)
**branch**: dev/training-poc

---

## 1. 状態確認

| 項目 | 結果 |
|------|------|
| `.env` NETKEIBA_COOKIE | ✅ 存在 (28 cookies、 nkauth 含む) |
| `tools/refresh_cookie.py --check` | ✅ ページアクセス OK |
| `data/cookies.json` (yt-dlp 用 JSON) | ❌ 不在 → 本 Session で生成 |
| `data/v18/videos_5_9/` 既存動画 | 0 件 (Session #52 は placeholder のみ) |

---

## 2. 本 Session の対応

`.env` の `NETKEIBA_COOKIE` 文字列 (Cookie ヘッダー形式) を `data/cookies.json` に変換。
JSON 形式は `tools/video_downloader.py` の `load_cookies_for_yt_dlp()` が yt-dlp 用 Netscape file に変換する入力。

実行: `python tools/v60_make_cookies_json.py`
出力: `data/cookies.json` (28 cookies、 5,770 bytes)

---

## 3. 5/9 重賞 race_id 確定 (`data/daily_predictions/20260509.csv` から)

| race | course | race_id | 距離 | 馬場 |
|------|--------|---------|------|------|
| 京都新聞杯 G2 (15:30) | 京都 | `202608030511` | 芝2200m | 良 |
| エプソムC G3 (15:45) | 東京 | `202605020511` | 芝1800m | 良 |
| 駿風S OP (15:20) | 新潟 | `202604010311` | 芝1000m | 稍 |

---

## 4. NEXT (Area B)

→ `tools/video_downloader.py` を **3 race_id を実値で** 実行
   (Session #52 の URL は PLACEHOLDER 文字列)

---

**Session #60 A 完了**
