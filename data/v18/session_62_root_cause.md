# Session #62 A: 動画 DL 失敗 root cause 詳細調査

**作成**: 2026-05-09 (Session #62 A、 5/9 朝 10:30+)
**目的**: Session #60 B / #61 で動画 DL 全 HTTP 400 失敗、 真の root cause 確定

---

## 1. 結論 (root cause)

**`race.netkeiba.com` および `db.netkeiba.com` が 5/9 朝、 全 client から HTTP 400 を返している (server-side block)**。

- Cookie / User-Agent / 全 client 種別とも無関係
- yt-dlp の問題ではなく **netkeiba の domain-level 制限**
- video DL の実装問題ではなく **取得元の物理的閉塞**

---

## 2. 検証 matrix

### 2.1 Server 側 reachability (5/9 11:23 JST)

| host | curl | python requests | Playwright real Chromium | 結果 |
|------|------|----------------|--------------------------|------|
| `www.netkeiba.com` | 200 | — | 200 | ✅ OK |
| `race.netkeiba.com` | **400** | **400** | **400** | ❌ 全 block |
| `db.netkeiba.com` | **400** | **400** | **400** | ❌ 全 block |
| `race.sp.netkeiba.com` | — | — | 200 | ✅ SP 版は OK (movie.html は 404) |
| `google.com` | 200 | — | — | ✅ network 正常 |

### 2.2 Header / Cookie variant (race.netkeiba.com)

| variant | status |
|---------|--------|
| Chrome desktop UA + cookies | 400 |
| Firefox UA + cookies | 400 |
| Safari iPhone UA + cookies | 400 |
| 空 UA | 400 |
| `curl/8.0.1` | 400 |
| `Googlebot/2.1` | 400 |
| Sec-Fetch-* + Referer + Accept-Language full | 400 |
| Cookie 完全付き (28 件 nkauth 含む) | 400 |
| Cookie なし | 400 |

→ **どの組合せでも 400**

### 2.3 Playwright (real Chromium, JS 実行可)

```
race.netkeiba.com/race/shutuba.html → status 400, body_len 39 (空)
race.sp.netkeiba.com/race/shutuba.html → status 200, body_len 232,056 ✅
race.sp.netkeiba.com/race/movie.html → status 404 (動画 page なし)
```

→ 真の Chromium でも 400。 **client-side でこれ以上の対応は不可能**。

---

## 3. yt-dlp specific verbose

```
yt-dlp 2026.03.17
ERROR: HTTP Error 400: Bad Request
```

netkeiba 用の **dedicated extractor は yt-dlp に存在しない** (generic 経由)。
generic extractor が page を取れない (400) ため、 video URL 抽出に至らない。

dedicated extractor があっても **server 400 では効かない**。

---

## 4. 推測される原因 (server side)

| 候補 | 確度 | 根拠 |
|------|------|------|
| 1. **maintenance window** | 高 | 5/9 (土) 朝 11:00 台、 race day 直前メンテで `race.*` のみ落ちる例あり |
| 2. **IP rate limit / abuse 検知** | 中 | Session #58 で 28 件 HTTP 429 (Discord 経由) → netkeiba 側でも同 IP に制限の可能性 |
| 3. **WAF / Cloudflare 強化** | 中 | sp 版は通る → 本家側だけ防御 |
| 4. **bot 検知 (TLS fingerprint)** | 低 | real Chromium も blocked、 fingerprint では説明つかず |

→ **#1 + #2 の合算が最有力**。 数時間〜半日で復旧する可能性あり。

---

## 5. 影響範囲

### 5.1 daily_predict.py

```python
url = 'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20260509'
→ status 400
```

→ 5/9 朝 8:00 の daily_predict は **すでに完走している** (今 cumulative_results.csv / daily_predictions/20260509.csv 存在)
→ 今 rerun すると同じく 400 で fail する状態

### 5.2 5/9 朝 9:30 race_auto_notify

→ 起動済 (race fetch は 8:45 完了済)、 timer は memory 内、 通知は schedule 通り発火
→ 影響は今後の **動画 DL のみ**

### 5.3 V15 投資

```
12R 1勝 ¥2,100 (案B改 strict) は 既に投票決定済
V15 model / predict_core / daily_predict 全不変
```

→ **5/9 投票には影響なし**

---

## 6. NEXT (Area B で確定)

→ B 領域で strategy 確定 (server 復旧 待ち or simulate fallback 強化)

→ どちらにせよ Session #62 は **simulate baseline + Playwright v2 framework 整備** で完成とする (server 復旧後に 1 行 rerun で実 features 取得)

---

**Session #62 A 完了 (root cause = netkeiba server-side 400 block、 client 側修復不可)**
