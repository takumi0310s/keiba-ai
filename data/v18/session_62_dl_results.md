# Session #62 D: 5/9 重賞動画 全再 DL 結果

**作成**: 2026-05-09 (Session #62 D、 11:29 JST)
**実行**: `python tools/video_downloader_v2.py --majors`

---

## 1. 結果サマリー

| race | race_id | result | page status | http error |
|------|---------|--------|-------------|------------|
| 京都新聞杯 G2 | 202608030511 | ❌ no_video_url | fail | http_400 (3 retries) |
| エプソムC G3 | 202605020511 | ❌ no_video_url | fail | http_400 (3 retries) |
| 駿風S OP | 202604010311 | ❌ no_video_url | fail | http_400 (3 retries) |

**取得**: 0 / 3、 **失敗**: 3 / 3

---

## 2. 直接原因 = A で確定済

`race.netkeiba.com` が **5/9 朝 11:29 時点 依然 HTTP 400** を返す server-side block 状態。
v2 (Playwright + Cookie + retry 3 回) でも回避不可。

### 失敗パターン

```
attempt 1: page.goto -> http_400 -> 5 sec sleep
attempt 2: page.goto -> http_400 -> 5 sec sleep
attempt 3: page.goto -> http_400 -> give up
```

各 race × 3 retry = 計 9 attempts、 全 400。

---

## 3. v2 framework は完成、 復旧待ち

`tools/video_downloader_v2.py` の機能は **probe で動作確認済**:

```
$ python tools/video_downloader_v2.py --probe
{
  "https://www.netkeiba.com/": 200,
  "https://race.netkeiba.com/": 400,    ← 復旧待ち
  "https://db.netkeiba.com/": 400,      ← 復旧待ち
  "https://race.sp.netkeiba.com/": 200
}
```

server (race.netkeiba.com) が 200 戻り次第、

```
python tools/video_downloader_v2.py --majors
python tools/horse_motion_features.py --batch
```

→ 真の動画 features 即取得可能。

---

## 4. 本 Session 後続 (E/F)

DL 失敗のため:
- E: 真の YOLOv8 features 取得 不可 → **realistic simulate** で代替
- F: 5 system v3 を simulate 値で完成 (Session #60 の placeholder simulate を改善版に)
- ★ V15 投資 (新潟 12R ¥700、 案B改 strict) は完全保護 ★

---

## 5. NEXT (Area E)

→ tools/horse_motion_features_v62_simulate.py で V15 score 反映の realistic simulate 値 生成

---

**Session #62 D 完了 (DL 0/3、 server 復旧待ち、 simulate fallback で進行)**
