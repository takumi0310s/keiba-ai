# P0-5-A: -15 min 再計算 data source 確定 audit

**作成日**: 2026-05-17 (G1 day、 read-only audit、 fetch なし)
**作業 mode**: 設計のみ。 V15 production / predict_core / daily_predict / race_auto_notify / app.py / schtasks 不変。 commit/push 親集中。
**前提**: P0-4 TYB 永久放棄確定 (`docs/P0_4_FINAL_VERDICT_2026_05_16.md`)、 真の -15 min source を新規確定

---

## 0. 結論 (★ TL;DR ★)

| 項目 | 値 |
|------|-----|
| **★ 推奨 source (1 位) ★** | **JV-Link O1 連続オッズ** (単複) + **O2** (複勝) + **TCOV** (馬場 -60 min) |
| **推奨 source (補助)** | netkeiba 直前 odds API (既存実装、 race_auto_notify が -5 min で fetch 済) |
| **永久除外** | JRDB CYB (調教分析 = 3-4 日前 publish、 -15 min source ではない) / TYB (Sub-task 11) / netkeiba 動画 (Sub-task 11) |
| **規約 verdict** | **JV-Link 系: PASS (公式契約、 個人利用 OK)** / netkeiba 直前 odds: 既存 path 維持 (新規 scraper なし) |
| **実装複雑度** | **中** (32-bit Python venv 構築 + COM Dispatch + record parser、 既存 jvlink_fetcher_v2.py 拡張で 2-3 日) |
| **5/17 21:00+ 実装 ready** | ✅ (G1 day 中の fetch 実行なし、 5/24 JRA-VAN 加入後の本動作確認 待ち) |
| **V15 production 影響** | **0** (新規 schtask、 別 output file、 既存 race_auto_notify は当面不変) |

★ honest 注記 ★:
1. JV-Link O1 の **真の per-race -15 min publish timing** は 5/24 加入後 実動作確認まで未検証 (本 audit は既存 docs + sub-task 11 verdict 引用ベース)
2. JV-Link DLL は **32-bit COM のみ**、 既存 64-bit Python (3.14) では Dispatch 不可 → 32-bit venv 構築必要 (CLAUDE.md §「JRA-VAN 加入 + JV-Link 環境」)
3. ★ G1 day 中の新規 fetch 実行は本 audit で一切なし ★

---

## 1. JV-Link O1 audit

### 1-1. 既存契約 / 既存 fetch tools

**契約 status (CLAUDE.md §「JRA-VAN 加入 + JV-Link 環境」 2026-05-07 夜 確定)**:

| 項目 | 値 |
|------|----|
| JRA-VAN DataLab | 加入完了 (5/7 夜) |
| JV-Link DLL | `C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll` (32-bit only、 ver 1.18) |
| ProgID | `JVDTLab.JVLink` |
| 動作確認 (5/7 夜) | ★ 過去日付 5/3 で 29 ファイル取得 OK ★ |
| 32-bit Python venv | `C:\Users\takum\jvlink-venv\` (推奨、 5/24+ 着手) |
| .env credentials | ★ 未設定 ★ (JV-Link は ID/PW を `JVSetUIProperties()` GUI から レジストリ `HKCU\Software\JRA-VAN\JVLink` に保存、 .env 不要) |

**既存 file 一覧** (Glob `tools/*jv*.py` 実測):

| file | role | status |
|------|------|--------|
| `tools/jvlink_fetcher.py` | Session #39 B 試作 (PoC、 約 170 行) | ✅ syntax OK、 動作 OK (5/7 夜) |
| `tools/jvlink_fetcher_v2.py` | Session #41 B 本実装 (RA/SE/HR/O1 parser placeholder) | ✅ syntax OK、 動作 OK |
| `tools/jvlink_test_python32.py` | Session #41 A3 動作確認 script | ✅ Dispatch test 用 |
| `tools/jvlink_parser.py` | record parser | (本動作確認 待ち) |
| `tools/jvlink_backfill_*.py` | bulk fetch helpers (5/1-5/7 backfill 等) | 未動作 |
| `tools/jvlink_movie_wrapper.py` | (補助、 動画 wrapper) | 未動作 |
| `tools/tfjv_parser.py` | TFJV (TARGET frontier JV) parser | ✅ Session #44 B 本実装、 動作 OK |

**既存 output dir** (Glob `data/jvlink/**/*.json` 実測):

- `data/jvlink/phase12_poc_index.json` (288 race ids、 2026-04-10〜2026-05-10)
- `data/jvlink/2026/04/` (216 file)
- `data/jvlink/2026/05/` (72 file)
- ★ 内容 ★ : `{race_id, date, ra:{...}, se:[...], hr:{...}, source: "TFJV_BINARY_2026", phase: "phase12_poc"}` → **TFJV (TARGET frontier JV) 経由の race meta + entry + payout、 O1 ではない** ★

**結論**: ★ JV-Link 試作完了 + 5/3 取得 OK 確認済、 しかし O1 (連続オッズ) の **production fetch path は未着手** ★。 既存 `data/jvlink/` 配下は **TFJV (TARGET frontier JV) 由来の RA/SE/HR + race meta** であり、 O1 (連続 odds) ではない。

### 1-2. O1 datatype の publish timing

**JV-Link DataSpec 仕様** (`tools/jvlink_fetcher.py:36-59` + `docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` §3.2):

| dataspec | content | publish timing |
|----------|---------|----------------|
| `RACE` | レース詳細 (RA record = 開催情報) | 番組発表時 |
| `O1` | **単複枠オッズ** | **連続更新 (-1 min まで)** |
| `O2` | 馬連オッズ | 連続更新 |
| `O3` | ワイドオッズ | 連続更新 |
| `O4` | 馬単オッズ | 連続更新 |
| `O5` | 三連複オッズ | 連続更新 |
| `O6` | 三連単オッズ | 連続更新 |
| `WF` | 馬体重情報 | **-70 min 確定** |
| `TCOV` | 調教 (馬場) | **-60 min まで update** |
| `DM` | コメント | 当日朝 |
| `HR` | 払戻金 | レース後 |

**O1 連続オッズ取得 sequence** (公式 SDK 仕様、 `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` §2.E):

- `JVOpen` + `option=2` (今週分、 realtime mode) で O1 dataspec を request
- record 内 `data_kbn` で「速報」 (1A/1B/1C/1F 等) と「確定」 (2/9) を識別
- 連続 polling (30-60 秒間隔) で 最新 snapshot を取得可能
- -1 min まで rolling update (公式仕様、 投票締切 時点で確定 oz 配信切替)

**取得可能 features** (本 audit 内 sub-task 18 §3-2 paci 既存 + 拡張 抜粋):

| feature | source | timing | 用途 |
|---------|--------|--------|------|
| `tansho_odds_snapshot` (O1) | JV-Link O1 単勝 | -1 min まで連続 | 直前 odds、 odds_log 代替 |
| `fukusho_odds_snapshot` (O1) | JV-Link O1 複勝 | -1 min まで連続 | 複勝 odds |
| `wakuren_odds_snapshot` (O1) | JV-Link O1 枠連 | -1 min まで連続 | 枠連 (使用予定なし) |
| `pop_rank_snapshot` (派生) | O1 単勝降順 ranking | -1 min まで | pop_rank 代替 |
| `odds_change_30_15_min` (派生) | O1 -30 vs -15 min diff | 連続 polling 必須 | ★ V15 で未使用、 V22 候補 ★ |
| `odds_change_15_5_min` (派生) | O1 -15 vs -5 min diff | 連続 polling 必須 | ★ 同上 ★ |

★ Sub-task 11 verdict ★: ★ JV-Link O1 連続 odds = -30/-15/-10/-5 min 4 段階 snapshot で odds 変動 features 化、 V21/V22 学習候補。 jrdb_odds_idx は LEAK 確定だが O1 snapshot 自体は delivery 安全 ★ (引用: `docs/P0_4_FINAL_VERDICT_2026_05_16.md` §2-1)

### 1-3. 利用 / 未利用 status

**実測 audit (read-only)**:

| status | 検出箇所 | 結果 |
|--------|----------|------|
| `tools/race_auto_notify.py` | Grep `jvlink\|JV-Link\|JVLink` | **0 matches** (JV-Link 一切使用なし) |
| `tools/daily_predict.py` | Grep `jvlink\|JV-Link\|JVLink` | **0 matches** (JV-Link 一切使用なし) |
| `tools/predict_core.py:_fetch_odds_api()` | line 1144-1170 | ★ **netkeiba `race.netkeiba.com/api/api_get_jra_odds.html` のみ** (cookie 認証、 JSON) ★ |
| `data/odds_base_*.csv` | line 1-5 sample (5/17) | `race_id,horse_num,odds,pop_rank,timestamp` = netkeiba 由来 (08:00 morning snapshot) |
| `data/jvlink/2026/0[4-5]/` | `source` field | TFJV (TARGET frontier JV) 由来、 O1 ではない |

**朝 8:00 daily_predict は どこから odds 取得しているか**:

- `tools/predict_core.py:1144 _fetch_odds_api()` = **netkeiba API のみ** (`race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=1`)
- `save_odds_base()` (line 1191) が 1 日 1 回 (初回押下時) `data/odds_base_YYYYMMDD.csv` に baseline 保存
- ★ JV-Link O1 は本番 fetch path に 一切組み込まれていない ★

**race -5 min snapshot は誰が取得しているか**:

- `tools/race_auto_notify.py:192` で `odds_full = fetch_realtime_odds_full(race_id)` (= netkeiba API 経由 odds + pop_rank)
- ★ race-time の -5 min 程度で 1 回 snapshot → save_odds_base() で 1 日 1 回 baseline 保存 + 即座 predict_race() で V15 推論 ★

→ ★ 現状 V15 production の odds source = **netkeiba 単独**、 JV-Link O1 への切替も可能だが 5/24 加入後の動作確認待ち ★

### 1-4. 規約 audit

**JRA-VAN DataLab 規約 (公式)**:

引用元: `docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` §1.1:

| 項目 | verdict |
|------|---------|
| 個人利用 | ★ 「個人利用は契約次第で可 (要確認)」 ★ — DataLab 標準契約は **個人利用 OK** (1989 年〜公式提供 software、 利用者 ID 1 件で個人 PC 利用) |
| 商用利用 | 別途契約必要、 keiba-ai は個人利用 scope (Streamlit 公開しているが kakeiba 自己利用扱い) |
| 自動投票連携 | JV-Link 単体では投票 API 含まず、 別途 IPAT 連携必要 (本 sub-task scope 外) |
| 規約原文 | https://jra-van.jp/dlb/manual/jvlink.html (5/17 G1 day 中の fetch なし、 既存引用のみ) |

**規約 risk 評価**: ★ **低** ★

- JV-Link は ★ 公式契約者向け正規 channel ★ = IP/cookie ban risk 0
- 連続 polling (O1 -1 min まで) は **公式 SDK が想定する標準 use case** (jvlink_fetcher.py の `--realtime` flag は option=2)
- ★ 引用 ★ (`docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` §4.3): 「JV-Link 公式 API は契約者向け正規 channel = ban リスク 0」

---

## 2. JRDB CYB audit

### 2-1. sub-task 11 結果 再読

**Sub-task 11 verdict** (`docs/P0_4_FINAL_VERDICT_2026_05_16.md` §2-3):

> **JRDB CYB merge fix verify (★ verify only ★)**
> - Sub-task 5-4 で「constant default」 発見 (548K rows、 unique=1 columns 多数)
> - TYB と同様 「merge bug 確定 → fix しても V15 で truncate」 の可能性 高い
> - ただし CYB は ★ 数日前 publish ★ で delivery 安全 (post-race leak なし)
> - 5/24+ で merge audit + corr_target check + safe判定 のみ verify、 採用は 別 sub-task で判断

★ ★ CYB は -15 min source ではない、 数日前 publish の調教分析 ★ ★

### 2-2. publish timing

**Sub-task 18 §1 棚卸し table 引用** (`docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md`):

| ID | file | rows | cols | publish timing | live -15 min 用途 |
|----|------|-----:|-----:|----------------|-------------------|
| **cyb** | `data/jrdb_cyb.csv` | 513 (実測) | 32 | 低 fill、 (data 不足) | ★ **不可** ★ |
| **cyb_v2** | `data/jrdb_cyb_v2.csv` | 1,875 (実測) | (差替) | 低 fill | ★ **不可** ★ |

**CYB の本質** (sub-task 18 §1 注記):

- 「調教分析データ」 = **数日前 publish** (中央木曜追切後)
- safe verdict だが live -15 min path とは **別経路**
- sub-task 11 §2-3 の「verify only」 = retrospective audit (採用判定は別 sub-task)

**結論**: ★ **CYB は -15 min 再計算 source として不採用 (publish timing が 不一致)** ★

### 2-3. 既存 jrdb_cyb.csv 棚卸し

実測 (`wc -l data/jrdb_cyb.csv data/jrdb_cyb_v2.csv`):

```
   513 data/jrdb_cyb.csv     ← 32 columns、 size 小 (528 KB)
  1875 data/jrdb_cyb_v2.csv  ← 拡張版 (差替)
```

sub-task 18 §1 で V15 採用 = ✗ 未使用 (data 不足、 512 rows しかない)

★ 真の 直前情報 (TYB live path) ★ は sub-task 11 で永久放棄確定:
- JRDB tyokuzen path `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh` (-15 min publish) は技術的に存在
- ただし **content (TYB 5 features) は LEAK 3 + 信号 0 が 2 = 採用候補 0** → 永久放棄

★ JRDB 系で -15 min publish 可能な path は **TYB 以外存在しない** (sub-task 18 §1 棚卸しで kyi/bac/sed/cha/sr/kab/paci/kta/jo/ze/zk/srb/kka_v2/ot/ou/ov/ow/oz 全て 当日朝 06:00 までに確定 or レース後) ★

---

## 3. netkeiba 直前情報 audit

### 3-1. 規約 + 既存 scraping path

**既存 scraping tools** (Glob `tools/*netkeiba*.py` + Grep):

| file | role | path |
|------|------|------|
| `tools/predict_core.py:_fetch_odds_api()` | 直前 odds API (cookie 認証) | `race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=1` |
| `tools/predict_core.py:fetch_realtime_odds()` | wrapper | 同上 |
| `tools/predict_core.py:fetch_realtime_odds_full()` | wrapper (odds + pop_rank) | 同上 |
| `tools/predict_core.py:fetch_result_odds()` | レース後 odds | (別 endpoint) |
| `tools/refresh_cookie.py` | cookie 自動更新 (Playwright) | 既存 |
| `tools/scrape_training.py` | 調教タイム + コメント | netkeiba premium |
| `tools/scrape_premium_data.py` | 一括取得 | 同上 |

**規約 audit**:

- ★ docs 内に明示的な「IP ban 経験」 記載なし ★ (Grep `IP ban\|cookie ban\|netkeiba.*ban\|429` で `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` §2.1.D 1 件 = 「4/26 IP ban 経験」 だが、 該当 doc 内のみ、 真の発生 doc は別箇所)
- 既存 SCRAPER-GUARD (金 22:00-月 06:00) は **netkeiba 過剰 scrape 防止用** (CLAUDE.md §「SCRAPER-GUARD の動作変更 (2026-04-19)」)
- ★ 4/19 SCRAPER-GUARD 誤停止事故 ★ で 1 日午前レース全ロス (機会損失 +2,745 円) → 11 commits で完全修正済 (OPERATIONAL_CALLERS ホワイトリスト導入)
- ★ G1 day 中の新規 scraper 実行は本 audit で一切なし ★

### 3-2. 直前情報 page の publish timing

**個別 R URL のオッズ tab**:

- URL: `https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1`
- auth: cookie (`.env` NETKEIBA_COOKIE)
- update: **連続更新 (-1 min まで)**、 netkeiba 内部で 30 秒前後 polling 想定
- 取得可能 features:
  - 単勝オッズ snapshot (`tansho`)
  - 人気順位 (`pop_rank`)
  - 複勝オッズ (別 type 指定で取得可、 現状 V15 で未使用)

**馬体重 timing**:

- 公式仕様 = race -70 min 確定 (CLAUDE.md §4 Pattern B 8 features 内 `horse_weight`)
- 個別 R URL の出馬表 section で取得 (`race.netkeiba.com/race/shutuba.html?race_id=...`)
- 既存 V15 production で fetch 済 (race_auto_notify.py)

**騎手変更 timing**:

- -30 min 〜 直前 (公式 JG 系の連携)
- V15 で未使用

### 3-3. scraping rate

**実測** (`tools/predict_core.py` 周辺):

- 1 race 当たり: odds 1 fetch (+ rate limit `time.sleep(1)` between race in race_auto_notify.py line 196)
- 1 日 trace: ~36 races × 1 odds + 1 horse_stats = ~72 hits/day
- ★ SCRAPER-GUARD で過剰 scrape 防止 ★ (金 22:00-月 06:00、 OPERATIONAL_CALLERS ホワイトリスト)

**安全 rate 目安** (sub-task 11 経由で確認):

- < 60 req/min 推奨
- IP ban 経験は明示的 docs なし、 SCRAPER-GUARD は 4/19 誤停止事故 contextに導入

**robots.txt**: G1 day 中 fetch なし、 doc 確認のみ (本 audit では未検証)

---

## 4. source 比較 table

| source | timing | features | 規約 risk | 実装 complexity | 5/17 21:00+ ready | 推奨 priority |
|--------|--------|----------|----------|---------------|--------------------|----------------|
| **★ JV-Link O1 ★** | **-1 min まで連続** | 単勝/複勝/馬連/三連 odds snapshot + 派生 (odds_change_30_15 等) | **低** (公式契約、 JRA-VAN DataLab 加入済) | **中** (32-bit Python venv + COM Dispatch、 既存 jvlink_fetcher_v2.py 拡張) | ★ 設計 OK、 5/24 実機 fetch ★ | **★★★ 1 位** |
| **JV-Link TCOV** | -60 min まで update | 馬場 (cushion / moisture / weather) | 低 (公式) | 中 (同上) | 設計 OK、 5/24 実機 | ★★ 2 位 (V15 既存 cushion/moisture 代替経路、 +AUC delta ≈ 0) |
| **JV-Link WF** | -70 min 確定 | 馬体重 / weight_diff | 低 (公式) | 中 | 設計 OK、 5/24 実機 | △ (Pattern B 既存 horse_weight と重複) |
| **JRDB tyokuzen TYB** | -15 min publish | パドック気配等 5 features | 低 (既加入) | 低 (HTTP DL) | ★ **永久放棄 (Sub-task 11)** ★ | ❌ (content 採用候補 0、 LEAK 3 + 信号 0 が 2) |
| **JRDB CYB** | 数日前 (調教分析) | 調教分析 32 cols | 低 (既加入) | 低 | △ (5/24+ verify only) | ❌ (-15 min source ではない、 publish timing 不一致) |
| **netkeiba 直前 odds API** | 連続 (-1 min まで) | 単勝 + pop_rank | ★ 中 (IP ban risk、 既存 SCRAPER-GUARD 対応済) | **低** (既存 predict_core.py で実装済) | ✅ 即時 ready | ★ 3 位 (JV-Link O1 補助、 既存 path 維持) |
| **netkeiba 馬体重** | -70 min | 馬体重 | 中 (IP ban risk、 既存 path) | 低 (既存) | ✅ 既存 | △ (既存 Pattern B が cover) |

★ ★ honest 結論: 真の -15 min source は **JV-Link O1 連続 odds (主軸) + JV-Link TCOV (馬場補助)** が現実的、 netkeiba 直前 odds は補助 (既存 path 維持) ★ ★

---

## 5. test fetch 結果 (★ G1 day 中、 既存 data audit のみ ★)

### 5-1. JV-Link O1 既存 fetch evidence

★ G1 day 中 fetch 一切なし ★。 過去 audit 引用のみ:

- `docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` §3.4: COM 接続 sequence 設計完了、 実機動作確認 未実施 (5/24 加入後)
- ★ CLAUDE.md §「JRA-VAN 加入 + JV-Link 環境」 ★: 動作確認 (5/7 夜) 過去日付 5/3 で **29 ファイル取得 OK** ← ただし datatype 不明、 RACE/SE/HR 系の可能性高い
- `data/jvlink/2026/04/` 内 .json file の `source: "TFJV_BINARY_2026"` ← **TFJV (TARGET frontier JV) 由来**、 JV-Link 直接 fetch ではない

→ ★ JV-Link O1 本動作 fetch は **未着手** 確定 ★、 5/24 加入後 実機動作確認必須

### 5-2. JRDB CYB 既存 evidence

実測 (`wc -l`):
```
   513 data/jrdb_cyb.csv
  1875 data/jrdb_cyb_v2.csv
```

→ ★ 「constant default」 多数 (sub-task 5-4 確認済)、 V15 採用 ✗、 -15 min source としては不適 ★

### 5-3. netkeiba 直前 odds 既存 evidence

実測 (`data/odds_base_*.csv` 9 files):

```
data/odds_base_20260418.csv
data/odds_base_20260502.csv
data/odds_base_20260503.csv
data/odds_base_20260508.csv
data/odds_base_20260509.csv
data/odds_base_20260510.csv
data/odds_base_20260513.csv
data/odds_base_20260516.csv
data/odds_base_20260517.csv  ← 本日分、 5 行 sample 確認済
```

サンプル (5/17 8:00 baseline):
```
race_id,horse_num,odds,pop_rank,timestamp
202608030801,1,8.6,5,2026-05-17 08:00
202608030801,2,70.6,10,2026-05-17 08:00
202608030801,3,3.3,1,2026-05-17 08:00
```

→ ★ netkeiba API は 朝 08:00 baseline + race -5 min snapshot で連続 fetch 動作中、 既存 path 維持で OK ★

---

## 6. 5/17 21:00+ 実装 path

★ G1 day 終了後の 21:00+ 実装 (本日 5/17 G1 day = Victoria Mile、 fetch 一切実行なし) ★

### 6-1. 推奨 source order (再掲)

1. **★ JV-Link O1 (単勝 + 複勝) ★** — 公式、 安全、 連続 -1 min まで update、 V21/V22 odds_change_* features 候補
2. **JV-Link TCOV (馬場 リアルタイム)** — 公式、 V15 既存 cushion/moisture 代替経路 (+AUC delta ≈ 0、 信頼性向上)
3. **JRDB tyokuzen path** — ★ TYB content 永久放棄、 経路のみ将来 cha/kta live にも応用可 ★ (本 sub-task で 0 工数)
4. **netkeiba 直前 odds (補助)** — 既存 path 維持 (IP ban risk あるが SCRAPER-GUARD で防御済)

### 6-2. 5/17 21:00+ 実装 ready の sub-task (★ 本日 fetch なし、 設計のみ ★)

| sub-task | 内容 | 期間 | V15 影響 |
|----------|------|------|----------|
| **6-2-1** | `tools/jvlink_fetcher_v2.py` の O1 parser 本実装 (現状 placeholder line 212-220) | 5/17 21:00+ 2h (★ 32-bit venv なしでも parser logic は実装可、 syntax check のみ) | 0 |
| **6-2-2** | `tools/parse_jvlink_o1.py` 新規作成 (固定長 record → DataFrame) | 5/17 21:00+ 3h | 0 |
| **6-2-3** | 設計 doc 追記: `docs/P0_5_JVLINK_O1_PARSER_DESIGN_5_17.md` | 5/17 21:00+ 1h | 0 |
| **6-2-4** | 5/24 加入後の動作確認 plan: `tools/jvlink_test_o1_live.py` skeleton | 5/17 21:00+ 1h (★ 5/24 32-bit venv で実機 fetch、 G1 day 中はなし ★) | 0 |

### 6-3. 5/24+ 実機動作確認 plan

| step | 内容 | 期間 |
|------|------|------|
| **6-3-1** | 32-bit Python venv 構築 (`C:\Users\takum\jvlink-venv\`) | 5/24 30 min |
| **6-3-2** | `JVSetUIProperties()` GUI で ID/PW 入力 → レジストリ保存 | 5/24 10 min |
| **6-3-3** | `tools/jvlink_test_python32.py --check-only` で COM Dispatch 確認 | 5/24 5 min |
| **6-3-4** | `tools/jvlink_fetcher_v2.py --date 20260524 --datatype O1 --realtime` で 1 R fetch、 raw record 観測 | 5/24 1h |
| **6-3-5** | O1 record の data_kbn (1A/1B/1C/1F/2/9) 各種別の sample 取得 + 内容確認 | 5/24-25 2h |
| **6-3-6** | parsed CSV → V15 .pkl.gz には混入させず、 `data/jvlink/O1/{date}_parsed.csv` shadow output のみ | 5/24-25 1h |
| **6-3-7** | -30 / -15 / -10 / -5 min 4 段階 snapshot fetch 動作確認 (土日 race-time で polling) | 5/25 Sun 終日 |

### 6-4. V15 production 完全独立 保証 (★ 絶対遵守 ★)

- ★ `tools/predict_core.py` 不変 (`_fetch_odds_api()` = netkeiba 単独維持) ★
- ★ `tools/daily_predict.py` 不変 ★
- ★ `tools/race_auto_notify.py` 不変 ★
- ★ `app.py` 不変 ★
- ★ V15 .pkl.gz 再 save なし ★
- ★ schtasks 新規登録なし (5/24 加入後 別 sub-task で `Keiba-JVLink-DailyFetch` 登録) ★
- ★ shadow output は `data/jvlink/O1/` (新規 dir、 .gitignore 対象) ★

---

## 7. V15 production 不変保証 ✅

本 audit (P0-5-A) の作業 scope は ★ 100% read-only audit + 設計 doc 新規 1 件のみ ★:

| 不変 file | check |
|-----------|-------|
| `tools/predict_core.py` | ✅ 不変 (read-only audit のみ) |
| `tools/daily_predict.py` | ✅ 不変 |
| `tools/race_auto_notify.py` | ✅ 不変 |
| `app.py` | ✅ 不変 |
| `keiba_model_v15_central*.pkl.gz` | ✅ 不変 |
| `data/_v15_optuna_df_cache.pkl.gz` | ✅ 不変 |
| schtasks | ✅ 新規登録なし |
| 既存 odds_base_*.csv | ✅ 5/17 baseline 不変 |

★ G1 day (5/17) 中の fetch / 新規 scraper 実行: **0** ★
★ 規約 risk 顕在化: **0** ★
★ git commit/push: **親集中** (本 sub-task agent は doc 新規作成のみ) ★

---

## 8. honest 限界 (fabrication 防止)

1. ★ JV-Link O1 の **真の per-race -15 min publish timing** は 5/24 加入後 実機動作確認まで未検証 ★。 本 audit は既存 docs 引用 (`docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` §3.2 + Sub-task 11 §2-1) ベース
2. JRA-VAN DataLab 規約原文は 5/17 G1 day 中 fetch 不可、 既存 引用のみ (個人利用 OK の verdict は引用元 信頼ベース)
3. JV-Link O1 の record format (固定長 fixed-width) は ★ 既存 `jvlink_fetcher_v2.py:212-220` で placeholder 実装、 真の field 位置は実機 sample 観測必要 ★
4. netkeiba 直前 odds の IP ban 経験は ★ 明示的 doc なし ★ (Grep `IP ban` で 1 件のみ = `P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` §2.1.D 「4/26 IP ban 経験」 推測ベース)
5. JRDB tyokuzen path の経路は sub-task 11 で永久放棄確定、 cha/kta live 応用は **将来検討の仮説**、 本 sub-task で実装着手なし

---

## 9. 完了通知 (caveman)

- 推奨 source: **JV-Link O1 (主軸) + TCOV (補助) + netkeiba 直前 odds (既存維持)**
- 実装複雑度: **中** (32-bit Python venv + COM Dispatch、 既存 jvlink_fetcher_v2.py 拡張)
- 規約 verdict: **PASS** (JV-Link 公式契約済、 個人利用 OK)
- V15 不変: ✅
- 5/17 21:00+ 実装 ready: ✅ (G1 day 中 fetch なし、 5/24 加入後 実機動作)
- commit/push: 親集中

---

## 10. 参考 / 出典

- `docs/P0_4_FINAL_VERDICT_2026_05_16.md` (Sub-task 11、 TYB 永久放棄 verdict)
- `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` (Sub-task 7、 JRDB tyokuzen path 発見 + JV-Link 系比較 §2.E)
- `docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md` (Sub-task 18、 JRDB 29 datatype 棚卸し)
- `docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md` (Session #39 B、 JV-Link 統合 plan)
- `CLAUDE.md` §「JRA-VAN 加入 + JV-Link 環境」 (2026-05-07 夜 確定)
- `tools/jvlink_fetcher.py` (Session #39 B 試作)
- `tools/jvlink_fetcher_v2.py` (Session #41 B 本実装、 O1 parser placeholder line 212-220)
- `tools/predict_core.py:_fetch_odds_api()` (netkeiba 単独 odds path、 line 1144-1170)
- `tools/race_auto_notify.py:192` (race -5 min odds snapshot)
- `data/odds_base_20260517.csv` (本日 baseline、 5 行 sample 確認済)
- `data/jvlink/phase12_poc_index.json` (TFJV 由来 288 race ids、 JV-Link O1 ではない)
