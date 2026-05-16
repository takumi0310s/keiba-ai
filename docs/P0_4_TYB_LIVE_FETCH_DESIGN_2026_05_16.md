# P0-4 TYB -15 min 直前 fetch 経路 設計

**作成**: 2026-05-16 Sat 19:55 JST
**作成 source**: 親 agent 指示 Sub-task 7 (read-only investigation + design 提案)
**作業 mode**: 設計のみ。 V15 production / cumulative_results.csv / predict_core.py / race_auto_notify.py 改変なし。 schtask 実登録なし。 git commit/push なし。
**前提**: P0-3 (Sub-task 5-1) で `data/Tyb/TYB*.lzh` の Last-Modified = 17:00 JST 確定 = race 終了後 publish → 朝 06:00 fetch では 404

---

## 0. 結論 (★ verdict ★)

| 項目 | 値 |
|------|-----|
| **★ 真の経路 ★** | **JRDB tyokuzen path: `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh`** |
| **★ 補助 経路 (live race trigger) ★** | **`http://www11.jrdb.com/nowracedata/data/{YYYY}/{YYYYMMDD}/now_racedata_json.json`** (Basic Auth 不要) |
| **実装可否** | ✅ **可能** (規約・技術 共に OK、 既存 JRDB 加入 で access 可) |
| **実装着手** | 5/18 (月) 以降 (まず paper shadow eval) |
| **工数** | **2-3 日** (fetch script + parser + schtask + shadow merge + docs) |
| **規約 risk** | ⚠ 低 (JRDB 既加入 member、 自動 fetch は JRDB 自社 software (Gold Generator) でも実装、 連続高頻度 polling は禁止 / 30-60 秒 polling は許容範囲) |
| **V15 production への影響** | ✅ **完全独立 0** (新規 schtask、 別 process、 別 output file、 既存 race_auto_notify は不変) |

★ 推奨理由 ★:
- 既存 `data/Tyb/TYB*.lzh` (17:00 JST publish) と **同一 file format / 同一 size 14319B** で **別 path に live 進行版** が存在 することを確認 (Sub-task 7 内 監査)
- `member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh` は **発走 15-20 分前 を目処に refresh される** (JRDB 公式 doc `jrdb_doc.pdf` §4.1 で明記、 5/2-5/16 の Last-Modified 5 週連続で 16:15-16:21 JST = 最終 race -15 min 観測)
- 既存 `tools/v21/jrdb_tyb_live_fetch.py` の 改修のみで実装可、 既存 parser (`tools/parse_jrdb_extended.py`) は流用可
- 規約: JRDB Gold Generator の "Tool → Option → 直前情報自動取得時間設定" 機能の HTTP backend を 自前 client で実装するだけ
- ★ honest 注記 ★: 5/16 の last-modified が **16:16 JST** であることは確認したが、 真の per-race -15 min 更新 (例: 14:50 race の 14:35 update) は シミュレーションで未確認。 5/17 (Sun) で polling 観測 必須

---

## 1. JRDB 公式 -15 min publish 経路調査 (Sub-task 7 監査結果)

### 1-1. 既存 path (失敗) — `/member/data/Tyb/`

| URL | Last-Modified (JST) | 解釈 |
|-----|---------------------|------|
| `http://www.jrdb.com/member/data/Tyb/TYB260516.lzh` | **17:03 JST** | 最終 race 終了後 publish (final 累積) |
| `http://www.jrdb.com/member/datazip/Tyb/2026/TYB260516.zip` | 17:04 JST | ZIP wrapper、 17:03 LZH と同じ |

→ ★ 朝 06:00 fetch 404 / 17:00 まで取得不可 / live use 不可 ★

### 1-2. ★ 新規発見 (live 経路) ★ — `/member/{YYYYMMDD}/tyokuzen/`

n_live JS (`/member/n_live_20260516.html` ← `/member/n_index.html` 経由) を逆解析。 `/member/{YYYYMMDD}/tyokuzen/` directory 配下に **live 進行版** TYB 含む。

```
http://www.jrdb.com/member/20260516/tyokuzen/
├── TYB260516.lzh         ← ★ live 進行 TYB (16:16 JST refresh) ★
├── 20260516.html          ← 全場 直前情報 HTML
├── t_e.html / t_w.html / t_l.html  ← 場別 直前情報 HTML (東京/京都/新潟)
├── tad_e.html / tad_w.html / tad_l.html  ← 場別 直前情報 詳細 HTML
├── p_p/padock_e.html      ← パドック詳細 HTML
└── 20260516ad.html        ← 直前情報 ad 版
```

5/2-5/16 (5 週連続) の Last-Modified 観測:

| 日付 | tyokuzen TYB.lzh | tyokuzen t_e.html | datazip TYB.zip |
|------|-------------------|--------------------|------------------|
| 5/2 Sat | 16:21 JST | 16:21 JST | 17:04 JST |
| 5/3 Sun | 16:16 JST | 16:16 JST | 17:01 JST |
| 5/9 Sat | 16:20 JST | 16:19 JST | 17:06 JST |
| 5/10 Sun | 16:18 JST | 16:18 JST | 16:54 JST |
| 5/16 Sat | **16:16 JST** | **16:15 JST** | 17:03 JST |

★ 全週 16:15-16:21 JST = 最終 race (16:25 発走) の **-10〜-5 min** に refresh 確認 ★。
HTML 内部に "1547時点のオッズ" "1616時点のオッズ" 等 inner timestamp、 "発走20分前更新" の note 明示。

### 1-3. 内容差確認 (同一 size 別 MD5)

```
$ md5sum data_tyb.lzh tyokuzen_tyb.lzh
6b5f2040b2bc7ecc7bf8fc18e13e7ed0 *data_tyb.lzh         # 17:03 JST 版
81dc80d37d021dc38daeba6ffc994730 *tyokuzen_tyb.lzh    # 16:16 JST 版
```

→ ★ size 14319B 同じだが content 別 = tyokuzen は **early snapshot**、 data/Tyb は **final** ★
→ tyokuzen を 各 race -15 min で polling すれば、 各時点の累積 TYB 取得可

### 1-4. JRDB 公式 doc 引用 (`http://www.jrdb.com/about/jrdb_doc.pdf` §4.1)

> 「直前情報」は、発走 20 分前前後を目処に発信しています。情報が古いまま・画面が変わらないといった時には、必ずリロード(再読み込み・更新)をしてからご覧になって下さい。

§4.1 はゴールドジェネレーター ユーザー向け説明だが、 backend 経路 (この設計が利用する path) は同一。

### 1-5. ★ Live race trigger ★ — `nowracedata` JSON (Basic Auth 不要)

n_live JS が 30 秒毎 polling する live trigger JSON を発見:

```
URL:  http://www11.jrdb.com/nowracedata/data/{YYYY}/{YYYYMMDD}/now_racedata_json.json
Auth: 不要 (member auth 無しで access 可、 cross-domain JSONP)
Size: ~587 B
Format: JSONP (now_racedata_callback(...))
```

content sample (5/16 19:54 JST 時点、 全 race 終了後):

```json
{
  "limit4_data": [   // 次 4 race
    {"datakbn":"5","hassotime":"1610","racekey":"05262712","txt_jo":"東京",
     "is_padock_comment":true, "is_rtn_comment":false},
    {"datakbn":"5","hassotime":"1630","racekey":"04261512","txt_jo":"新潟",
     "is_padock_comment":true, "is_rtn_comment":false}
  ],
  "now_data": {     // 現在 active race (-15 min trigger)
    "now_racekey":"04261512","now_jo":"新潟","now_hassotime":"1630"
  },
  "now_rtn_data": [],
  "ts":"20260516071453"  // last update timestamp
}
```

★ 30-60 秒 polling で `now_data.now_hassotime` が更新 → 各 race -15 min trigger 検出可能 ★

### 1-6. JRDB API / RSS / Atom feed

なし。 すべて HTTP GET の static / dynamic file fetch。

---

## 2. 代替 source 比較

| source | -15min 可? | features cover | 規約 | 工数 | 推奨度 |
|--------|-----------|----------------|------|------|--------|
| **★ JRDB tyokuzen path ★** | ✅ | full TYB (17 features) | OK (既加入) | S (2d) | **★★★ 最推奨** |
| JRDB datazip 17:00 (現状) | ❌ race 後 | full TYB | OK | 0 | 不可 (確定済) |
| JRDB tyokuzen HTML (parser) | ✅ | partial (HTML 解析) | OK | M (3d) | 第 2 候補 (LZH 不安定時) |
| `nowracedata_json.json` | ✅ (trigger のみ) | flag info のみ (padock/rtn 有無) | OK | XS (0.5d) | ★ trigger 補助、 単独不可 |
| **netkeiba パドック評価** | ✅? (-30 min?) | partial (padock_score、 odds は別 API) | ⚠ scrape risk (cookie ban 経験あり) | M (3d) | 第 3 候補 |
| JRA-VAN JV-Link `WF` | ✅ -70 min | 馬体重 / weight_diff のみ (TYB 17 features の 2/17 cover) | OK | M (3d、 32-bit Python venv 構築) | 部分代替 |
| JRA-VAN JV-Link `DM` | ✅ | コメント文字列 (TYB index は cover しない) | OK | M | 補助 |
| JRA-VAN JV-Link `TCOV` | ✅ -60 min まで update | 馬場 (cushion / 含水 / weather) | OK | M | TYB の baba_code 代替 のみ |
| JRA-VAN JV-Link `O1` | ✅ 連続更新 (-15 min OK) | 単複オッズのみ (TYB tansho_odds と 同等) | OK | M | tansho_odds 代替可 |
| JRA 公式 (race.jra.go.jp) | ✅ (馬体重等) | 馬体重 / 馬場 / 騎手変更 のみ | OK | L (HTML parse 複雑) | 不推奨 |
| TARGET frontier JV | ✅ | full JV データ | OK | L (TARGET TFJV bridge) | 不推奨 (TARGET 環境前提) |

### 2.1 各代替 source 詳細

#### A. JRDB tyokuzen path (★ 最推奨 ★)

- **fetch URL**: `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh`
- **auth**: HTTP Basic (.env `JRDB_ID` / `JRDB_PASSWORD` 既存)
- **features cover**: ✅ full TYB 17 features (既存 parser `tools/parse_jrdb_extended.py` で parse 可)
- **update freq**: 1 race 終了後 + 次 race -20 min に refresh (推定、 5/17 実観測 必須)
- **規約**: JRDB Gold Generator (自社 software) が同じ backend を 自動取得する 設計、 60 秒 polling は許容範囲 (要 JRDB FAQ 再確認)
- **失敗時 fallback**: data/Tyb/TYB*.lzh (17:00 final version) に切替

#### B. nowracedata JSON (★ trigger 補助 ★)

- **fetch URL**: `http://www11.jrdb.com/nowracedata/data/{YYYY}/{YYYYMMDD}/now_racedata_json.json`
- **auth**: 不要
- **features cover**: ❌ (only trigger info: 次 race の hassotime / padock_comment 有無 flag)
- **update freq**: 30-60 秒 (JSONP polling 設計)
- **用途**: race -20 min 検出 → TYB tyokuzen fetch trigger
- **規約**: JSONP cross-domain なので polling 制限ゆるい (要 監視)

#### C. JRDB tyokuzen HTML parser (★ LZH 不安定時 第 2 候補 ★)

- **fetch URL**: `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/t_{e,w,l}.html`
- **auth**: HTTP Basic
- **features cover**: partial (~12/17 features、 Shift-JIS HTML parse 必要)
- **update freq**: TYB.lzh と同じ (同時 refresh)
- **用途**: LZH parse 失敗時 fallback

#### D. netkeiba パドック評価 (★ 第 3 候補 ★)

- **fetch URL**: `https://race.netkeiba.com/race/shutuba.html?race_id={race_id}` の パドック section
- **auth**: 既存 NETKEIBA_COOKIE
- **features cover**: padock_score / weight (TYB 17 features の 3/17)
- **規約 risk**: ⚠⚠ 高 (4/26 IP ban 経験、 1 fetch あたり 5-10 秒 cooldown 必須)
- **不採用理由**: JRDB tyokuzen で full features OK のため代替不要

#### E. JRA-VAN JV-Link 系 (部分代替、 並走可)

| dataspec | -15 min OK? | cover | 工数 |
|----------|-------------|-------|------|
| `WF` 馬体重 | ✅ (-70 min) | weight_diff / horse_weight | M |
| `O1` 単複オッズ | ✅ (連続) | tansho_odds / fukusho_odds | M |
| `DM` コメント | ✅ | text data (encode 困難) | M |
| `TCOV` 馬場 | ✅ (-60 min) | cushion / moisture (V15 で既存) | M |

- **不採用理由**: JRDB tyokuzen TYB で 17 features 全 cover、 JV-Link 個別 fetch の総工数 が高い
- ただし P1+ で **JV-Link を主軸に切替** する候補 (JRDB 規約 risk 顕在化時)

#### F. 既存 schtask 延長 (★ 不可 ★)

- DailyPremiumScrape は AM3:00 schtask、 02:55 で停止
- -15 min は AM 3:00 と 完全別タイミング、 別 schtask 必須

---

## 3. fetch timing 設計

### 3.1 schedule (新規 schtask)

★ 推奨 ★ 段階 1: **shadow eval (5/18 Sun 〜 5/24 Sat、 30R 集積)**

| schtask | trigger | action |
|---------|---------|--------|
| `Keiba-TybLiveFetch_Sat_0900` | 土 09:00 JST | `tyb_live_fetch.py --watch-mode` start (background)、 race 終了 まで polling |
| `Keiba-TybLiveFetch_Sun_0900` | 日 09:00 JST | 同上 (Sun) |

`tyb_live_fetch.py --watch-mode` 動作:

```
loop (poll every 60s):
  1. fetch nowracedata_json.json (no auth)
  2. parse now_data.now_hassotime
  3. if now_hassotime change から 5 分以内 (= race -15 min 帯):
     - fetch tyokuzen TYB.lzh (with auth)
     - if Last-Modified changed (vs 前回):
       - save to data/jrdb_tyb_live/{YYYYMMDD}/TYB_{HHMM}.lzh
       - parse → data/jrdb_tyb_live/{YYYYMMDD}/parsed_{HHMM}.csv
       - log to data/jrdb_tyb_live/fetch_log.csv (date, time, race_key, last_mod, sha256)
  4. exit when 全 race 終了 + 30 分経過 (= 17:30 JST)
```

★ 推奨 schedule (V15 production への影響 = 0 を絶対遵守) ★

| timing | event | output |
|--------|-------|--------|
| race -20 min | nowracedata から trigger 検出 | log |
| race -15 min | tyokuzen TYB.lzh fetch | new file (shadow only) |
| race -10 min | parse + merge | live_tyb_features.csv (shadow) |
| race -7 min | (★ 段階 2 で実装、 shadow 通知 ★) | Discord #updates (TYB-aware paper prediction) |
| race -5 min | 既存 race_auto_notify (V15 通常 通知) | 既存 #買い目 (TYB 未使用) |

### 3.2 既存 race_auto_notify への影響 (★ 完全独立 0 ★)

- 新規 schtask `Keiba-TybLiveFetch_*` は **別 process / 別 script** で動作
- 既存 `tools/race_auto_notify.py` は **完全不変** (5/8 修正 commit 維持)
- 既存 `race_auto_notify` 内の `fetch_jrdb_tyb()` は 朝 fetch だが、 5/16 確認 で 404 = silent failure 中、 ★ 既存 behavior に影響なし ★
- ★ shadow eval 30R 後 採用判定 → 段階 2 で `predict_core.py` への TYB merge 検討 (この時 production code 改修、 慎重に GO/NO-GO 判断) ★
- ★ 通知 ★ shadow eval 初日 は **#updates channel のみ** (誤運用 防止)、 production 通知 (#買い目) には絶対混入させない

### 3.3 SCRAPER-GUARD との conflict

- 既存 SCRAPER-GUARD は金 22:00-月 06:00 (netkeiba 過剰 scrape 防止用)
- 新規 schtask は **Sat 09:00 / Sun 09:00** = SCRAPER-GUARD 解除後
- ただし: tyb_live_fetch は `OPERATIONAL_CALLERS` ホワイトリストに 追加必要 (将来 平日 nightly や 月-Mon 03:00 早朝 sanity check で 干渉 防止)

---

## 4. 実装 risk 評価

### 4.1 fetch 失敗時 fallback

| failure | fallback |
|---------|----------|
| tyokuzen TYB 404 (race 前 で publish なし) | 30 秒後 retry、 最大 5 回 → 諦めて NO_TYB |
| Last-Modified が動かない (1 時間以上) | log warning、 NO_TYB 確定 |
| LZH 解凍失敗 | last_known 良好 LZH を 採用 (前 race の cumulative TYB) |
| Basic Auth 401 | log critical、 Discord 通知 → 手動介入 (`.env` 確認) |
| nowracedata JSON 404 | 30 秒後 retry、 5 回 失敗で 5 分 sleep ループ、 ★ TYB fetch は 別経路 で続行 ★ |

NO_TYB fallback:
- ★ V15 単独運用 (既存 race_auto_notify 既存挙動 と完全一致) ★
- shadow output は "NO_TYB" record で 保存 (5/9 fetch 停止 時と同じ pattern)

### 4.2 network 障害時 retry logic

- 各 fetch は **3 回 retry (3 秒 sleep)**、 失敗は log → 次 race へ
- timeout 30 秒 (既存 download_jrdb.py と同等)
- ★ 復旧 ★ 既存 schtask で **毎 60 秒 polling 再開**、 自動回復

### 4.3 規約 risk (JRDB scrape)

- ★ 規約原文確認 必要 (5/18 必須 step) ★
  - `http://www.jrdb.com/kiyaku.html` で 詳細 (TLS cert 問題で WebFetch 不可、 5/18 手動確認)
- ★ 観測 推奨 polling 頻度 ★: **60 秒** (n_live JS は 30 秒だが 安全側で 2 倍に)
- 1 race 当たり fetch 数: ~10 polls (-20 min から -10 min) = 1 race 内で nowracedata 10 hits + TYB.lzh 1-2 hits (Last-Modified change 時のみ download)
- 1 日 trace 量: 12 races × 12 polls = 144 nowracedata hits + 12 TYB.lzh fetch = ~165 hits/day = JRDB Gold Generator 想定 traffic と同等
- ★ 連続 fetch 禁止 ★: `time.sleep(60)` 厳守、 30 秒未満 polling は ban risk

### 4.4 既存 schtasks との priority/conflict

- 5/8 DailyPremiumScrape AM3:00 (土日早朝早特例): 干渉なし (時間帯違う)
- Sat 0845 race_auto_notify: 干渉なし (別 script)
- Mon 03:00 早朝 sanity check: 干渉なし (週末のみ schtask)

### 4.5 ★ data leak risk (★ critical) ★

- ★ shadow output は production に絶対 inject しない ★
  - `data/jrdb_tyb_live/` (新規 dir、 `.gitignore` 対象)
  - `predict_core.py` は `jrdb_tyb_live/` を参照しない
  - 5/16 確認 で V15 .pkl.gz は TYB feature を 学習 data として 使用していない
- ★ shadow 通知も独立 channel (#updates) のみ ★
- ★ shadow eval 30R 集積 後の判定 ★: P0-3 で TYB content leak は 確認済 (✅ 安全)、 +AUC delta = +0.1429 (5CV、 retrospective)、 live -15 min snapshot で同 delta 出れば production 採用候補

---

## 5. 5/18+ 実装 plan

### 5.1 段階 1 (5/18 Mon〜5/19 Tue): fetch script 実装

| step | 内容 | 工数 |
|------|------|------|
| 5-1 | `tools/v21/jrdb_tyb_live_fetch.py` 改修: tyokuzen path 追加、 `--watch-mode` 実装 | 4h |
| 5-2 | `tools/v21/jrdb_tyb_live_fetch.py`: nowracedata polling + race trigger 検出 | 3h |
| 5-3 | `tools/parse_jrdb_extended.py` 流用 で TYB parse → `data/jrdb_tyb_live/{date}/parsed_{HHMM}.csv` | 2h |
| 5-4 | unit test: 過去 5/9 tyokuzen TYB を mock fetch、 parse 正常確認 | 2h |
| 5-5 | dry-run: 5/19 (Tue) 平日 = 非開催日 で `--dry-run` 実行、 endpoint 動作確認 (404 OK 期待) | 1h |

### 5.2 段階 2 (5/20 Wed〜5/22 Fri): schtask 登録 + 5/23 Sat shadow eval

| step | 内容 | 工数 |
|------|------|------|
| 5-6 | `tools/register_tyb_live_fetch_schtasks.ps1` (新規)、 dry-run 確認 | 2h |
| 5-7 | shadow eval 設計: `data/jrdb_tyb_live/shadow_pred_5_23.csv` 出力 (V15 prediction + TYB-aware paper prediction の 2 column 並走) | 3h |
| 5-8 | `tools/notify.py` を 拡張、 shadow 通知 専用 channel (Discord #updates) で TYB-aware 結果のみ通知 | 1h |
| 5-9 | nightly_sanity_check 拡張: 翌日 schtask に TybLiveFetch を含める | 1h |

### 5.3 段階 3 (5/23 Sat〜5/24 Sun): live shadow 観測

| step | 内容 | 工数 |
|------|------|------|
| 5-10 | 5/23 Sat / 5/24 Sun 実観測: tyokuzen TYB の per-race -15 min refresh を 30 R で確認 | passive (運用観察) |
| 5-11 | observation report: `docs/P0_4_TYB_LIVE_OBSERVATION_5_24.md` | 2h |

### 5.4 段階 4 (5/25 Mon〜): GO/NO-GO 判定

判定基準:
- ✅ GO 条件 1: 30R 中 25+ で TYB tyokuzen が -15 min から -5 min の間に refresh 観測
- ✅ GO 条件 2: parsed TYB の 17 features 完整性 ≥ 95%
- ✅ GO 条件 3: V15+TYB paper prediction の win rate が V15 単独 比 +2pt 以上
- ✅ GO 条件 4: JRDB 規約 観点 で 60 秒 polling が許容範囲 と確認

GO 達成時 → P1+ で `predict_core.py` への TYB merge 検討 (★ V15 .pkl.gz 改修 / 再学習 は 別 issue、 段階 5 で慎重 判定 ★)

### 5.5 段階 5 (★ 別 issue、 6 月以降 ★): V22 retrain or V15+TYB hybrid

- V22 retrain (TYB を 学習 data に 含めた版) は P1 issue
- V15+TYB hybrid (V15 prediction + TYB calibrator overlay) は P0.5 issue
- ★ P0-4 (この設計) では shadow eval まで のみ ★

---

## 6. fallback / 失敗時 対応

### 6.1 NO_TYB fallback (現状の V15 と同等)

- live fetch 失敗 → `data/jrdb_tyb_live/{date}/NO_TYB_RACE_{race_key}.txt` を 0 byte で touch
- shadow merge は NO_TYB row を 行ごと skip (V15 prediction のみ)
- ★ production race_auto_notify は影響なし (NO_TYB は default) ★

### 6.2 network 障害 retry

- nowracedata 30 秒 polling、 失敗 5 回 → 5 分 sleep → 再開
- TYB fetch 3 回 retry (3 秒 sleep)、 失敗 → 60 秒後 再試行
- 30 分連続 失敗 → Discord 通知 (#updates) → 手動介入

### 6.3 規約変更時 immediate stop

- JRDB 規約 改訂 (`http://www.jrdb.com/kiyaku.html`)、 IP ban、 401 unauth が観測されたら:
  - schtask 即停止 (`schtasks /End /TN Keiba-TybLiveFetch_*`)
  - JV-Link 代替経路 (§2.E) に 切替検討
- ★ shadow output は git に commit しない (`.gitignore` 対象) ★

---

## 7. 代替案: P0-3.5 (odds_base only sub-model)

P0-3 監査 で odds_base (08:00 morning snapshot) のみで AUC delta 検証 を提案。 本 P0-4 plan に対する **代替案 / 並走候補**:

| 案 | 工数 | data 確実性 | AUC delta |
|-----|------|------------|-----------|
| **P0-4 (本案、 TYB live)** | 2-3 日 | ⚠ refresh timing 観測必要 (5/23 確認) | +0.1429 (P0-3 retrospective、 真の値 live 検証 待ち) |
| **P0-3.5 (odds_base only)** | 1 日 | ✅ 既存 odds_base (08:00) で 即時 確認可 | 推定 +0.05 〜 +0.10 (TYB log-corr 0.66 から estimate) |

★ 推奨 ★: 並走 (P0-4 と P0-3.5 を 同日 5/18 着手、 早い方 優先)
- P0-3.5 が **AUC delta ≥ 0.05** で OK なら → P0-3.5 を先行 deploy、 P0-4 は P1 へ後回し
- P0-3.5 が **AUC delta < 0.03** で 不足なら → P0-4 を fast-track

---

## 8. honest 限界

1. **★ 5/16 観測 1 回のみ ★**: tyokuzen TYB の per-race -15 min refresh は **過去 5 週連続の Last-Modified パターン** で間接的に裏付けたが、 ★ live polling での per-race 単位の refresh 観測 は 5/23 Sat まで未確認 ★
2. **JRDB 規約 ★ 原文未読 ★** (TLS cert 問題で web fetch 失敗)、 5/18 手動 confirm 必須
3. **nowracedata JSON ★ no-auth path ★** が将来 auth 必須化される可能性 (JRDB 仕様変更 risk)
4. **HTML/LZH parse の Shift-JIS / CP932 mixing**: 既存 parser でカバーされているが、 tyokuzen 版で format 微差 が ある可能性 (1-2 features 取得失敗 risk)
5. **fetch_timing race condition**: 11R と 12R の間隔が 25 分しかない場合、 -15 min trigger が 重複 する可能性 (logic で除外 必要)
6. **★ JV-Link 代替経路の動作確認 0 ★**: §2.E の JV-Link 経路は 5/15 unlock 後 production fetch 未着手、 fallback として動かない 可能性 ある

---

## 9. 重要 警告

- ★ 5/16 現在 commit 上の `data/tyb_top3_predictor.pkl` (b4948d6a) を 本 fetch 経路の output で 即座に live deploy しないこと ★
  - P0-3 監査 で content leak は ✅ 安全 確認済、 真の delivery 経路 (本 P0-4) で 30R shadow eval まで NO-GO
- ★ V15 production への影響 ★: 0 (新規 schtask、 別 process、 別 output)
- ★ git commit/push なし (Sub-task 7 は 設計 のみ) ★
- ★ schtask 実登録なし (`tools/register_tyb_live_fetch_schtasks.ps1` は 段階 2 で 別 commit) ★

---

## 10. 参考 / 出典

- **JRDB 公式 doc**: `http://www.jrdb.com/about/jrdb_doc.pdf` §4.1 (発走 20 分前 更新)
- **JRDB n_live JS reverse-eng**: `http://www.jrdb.com/member/n_livejs/n_live51_for_dev.js` (Sub-task 7 内 監査済)
- **観測 raw data**: 5/16 Sat 19:54 JST 時点、 `data/tyb_publish_log.csv` (既存) + Sub-task 7 内 curl 観測
- **既存 P0-3 監査 report**: `docs/TYB_LEAK_AUDIT_2026_05_16.md`
- **既存 TYB fetch tool**: `tools/v21/jrdb_tyb_live_fetch.py` (改修対象)
- **既存 TYB parser**: `tools/parse_jrdb_extended.py` (流用)
- **既存 publish monitor**: `tools/tyb_publish_monitor.py` (data/Tyb 17:00 path 専用、 tyokuzen path 未対応)

### 観測 raw output (再現可能性確認用)

```bash
# tyokuzen TYB.lzh Last-Modified
$ curl -sI -u "$JRDB_ID":"$JRDB_PW" \
  http://www.jrdb.com/member/20260516/tyokuzen/TYB260516.lzh
HTTP/1.1 200 OK
last-modified: Sat, 16 May 2026 07:16:12 GMT  # = 16:16 JST
Content-Length: 14319

# data/Tyb path Last-Modified (比較用)
$ curl -sI -u "$JRDB_ID":"$JRDB_PW" \
  http://www.jrdb.com/member/data/Tyb/TYB260516.lzh
HTTP/1.1 200 OK
last-modified: Sat, 16 May 2026 08:03:30 GMT  # = 17:03 JST
Content-Length: 14319

# Content差 (同 size 別 MD5)
$ md5sum data_tyb.lzh tyokuzen_tyb.lzh
6b5f2040b2bc7ecc7bf8fc18e13e7ed0 *data_tyb.lzh         # 17:03 final
81dc80d37d021dc38daeba6ffc994730 *tyokuzen_tyb.lzh    # 16:16 live snapshot

# nowracedata JSON (auth 不要)
$ curl -sS http://www11.jrdb.com/nowracedata/data/2026/20260516/now_racedata_json.json
now_racedata_callback({"limit4_data":[...], "now_data":{"now_racekey":"04261512",
"now_hassotime":"1630"}, ...})
```
