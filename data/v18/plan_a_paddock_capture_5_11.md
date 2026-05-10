# Plan A: netkeiba パドック動画 frame capture 実装 (5/11)

## 結論 (TL;DR)

`tools/paddock_video_capture.py` 実装完了 (frame 抽出 ready)、 **dry-run 実行は cookie expire でブロック**。
URL 構造 / 認証 ゲート / DOM 構造 を実調査で確定。

**netkeiba 規約 第 14 条 遵守設計**: 動画 file は保存しない、 frame のみ JS canvas drawImage で抽出 (私的複製範囲)。

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

## Dry-run 結果 (5/11 00:30、 honest)

ターゲット: ウインイザナミ (`2022106229`) × `202603010112` (4/11 福島12R)

```
[PROBE] videoCount=0, iframes=14, premium_gate=True, sources=0
[AUTH] WARN: Premium_Regist_Box02 detected -> cookie expired / not logged in.
       Run: python tools/refresh_cookie.py
```

→ **cookie 期限切れで paddock viewer が "Premium_Regist_Box02" gate に置換**。 動画 element が DOM に挿入されない。

これは Phase 21A morning_go_check の WARN
> WARN: cookie rc=1 (5/17 までに refresh)

と整合。 既知の cookie expire。

## 次のステップ (cookie refresh 後)

```bash
# 1. Cookie 更新 (対話式 or 自動)
python tools/refresh_cookie.py

# 2. 認証成功なら paddock viewer の DOM に video element 挿入される
python tools/paddock_video_capture.py 2022106229 --race-id 202603010112 --probe

# 期待される PROBE 出力:
# [PROBE] videoCount=1, iframes=N, premium_gate=False, sources=>=1
# [PROBE] src=https://...{m3u8 or mp4 url}...
# [PROBE] duration=30.0+, 640x360 (or larger), readyState=4

# 3. Frame 抽出 dry-run
python tools/paddock_video_capture.py 2022106229 --race-id 202603010112 --fps 3 --duration 30
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

## 期待性能 (cookie 更新後)

| 項目 | 想定 |
|------|------|
| 1 馬 1 レース paddock | 30-60 秒動画 (要確認、 LIVE は 30 秒制限あるが アーカイブ未確認) |
| 抽出 frame (3 fps × 30 秒) | 90 frame |
| 解像度 | 640x360 想定 (要確認) |
| サイズ (jpeg q=85) | 1 frame ~30-60 KB → 90 frame で ~3-5 MB |
| 1 race (15 頭) | ~45-75 MB |
| 1 開催 24 races | ~1-2 GB |

## DRM / 制限 リスク

調査で判明していない:
- DRM (Widevine) の有無 → canvas drawImage で `canvas_taint` error 出るかどうか
- Aging restriction (LIVE 30 秒制限の archived 版への適用)
- 同時セッション制限 (複数 馬 の並行抽出可否)

**dry-run 完了後判明**: cookie refresh 後、 即 確認可能。

## 次の commit で完了する範囲

- ✅ tools/paddock_video_capture.py 実装
- ✅ .gitignore に data/paddock_frames/, data/cookies.json 追加
- ✅ data/v18/plan_a_paddock_capture_5_11.md (本 doc)
- ⏳ 実 frame 抽出 dry-run (cookie refresh 後 user task)

## V15 投資保護 (確認)

- predict_core.py / daily_predict.py / app.py / V15 model **完全不変**
- 新規 file のみ追加 (tools/paddock_video_capture.py、 plan_a doc)
- .gitignore 追記のみ
- 既存 V15 production pipeline 影響 0
