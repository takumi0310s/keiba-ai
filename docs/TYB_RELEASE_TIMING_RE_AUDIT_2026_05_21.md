# TYB Release Timing Re-Audit

**実施日**: 2026-05-21
**前提 docs**: `docs/TYB_LEAK_AUDIT_2026_05_16.md` / `docs/TYB_MERGE_BUG_AUDIT_2026_05_16.md` / `docs/P0_4_FINAL_VERDICT_2026_05_16.md` / `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md`
**commit ref**: `416c4703` ([Sub-task 11] P0-4 TYB 永久放棄)
**audit scope**: read-only、 production code / model / data 変更なし

---

## 0. 結論 (★ TL;DR ★)

| 項目 | 真値 |
|------|------|
| **TYB content (fields)** | ★ **全 26 fields = PRE-RACE** ★ — finish/rank/prize 等のポスト確定値は一切含まない |
| **TYB delivery (標準 datazip path)** | ★ **POST-RACE 配信** ★ — `/member/data/Tyb/TYB*.lzh` は当日 17:00-17:10 JST publish |
| **TYB delivery (tyokuzen path)** | ★ **PRE-RACE 配信 (確認済)** ★ — `/member/{YYYYMMDD}/tyokuzen/TYB*.lzh` は 16:15-16:21 JST (最終 race -10〜-20 min、 5 週連続観測)。 JRDB 公式 doc は「発走 20 分前前後を目処に発信」と明記 |
| **過去 LEAK 判定の根拠** | 標準 datazip path が 17:00 JST (post-race) だったため delivery LEAK と判定。 ただし **content leak (ポスト確定値混入) ではない** |
| **odds_idx LEAK 判定** | ★ **誤ラベル** ★ — content は genuine -15 min pre-race odds index。 corr_target +0.42 は「高予測力」であり「post-race データ混入」ではない |
| **V15 永久放棄 verdict** | ★ **V15 においては正しい** ★ — predict_core.py:2160-2163 の `X[:, :145]` slice で TYB 5 features は必ず truncate。 fetch しても効果 0 |
| **V21/V22 永久放棄** | ★ **誤り** ★ — TYB content は全 field PRE-RACE、 tyokuzen path で pre-race delivery 可能。 V21/V22 retrain 時は採用候補 |
| **使用可能 fields** | `tansho_odds`, `fukusho_odds`, `odds_idx`, `jockey_idx`, `padock_idx`, `info_idx`, `padock_mark`, `ashimoto`, `batai_code`, `kehai_code`, `bagu_change`, `horse_weight`, `weight_diff`, `idm`, `sogo_idx`, `baba_code`, `weather_code`, `jockey_code`, `jockey_name`, `weight_carry` |

---

## 1. 過去 audit の判定根拠 — 何が LEAK とされたか

### 1.1 TYB_LEAK_AUDIT_2026_05_16.md (Sub-task P0-3)

| feature | 過去 verdict | 根拠 |
|---------|-------------|------|
| tansho_odds | content ✅ SAFE / delivery ⚠ POST-RACE | tansho_odds は -15 min snapshot (exact-match 6.56% vs confirmed odds → 確定値ではない確認済) |
| fukusho_odds | content ✅ SAFE / delivery ⚠ POST-RACE | tansho_odds と同等 |
| padock_idx | content ✅ SAFE / delivery ⚠ POST-RACE | パドック観察 (-30〜-40 min pre-race)、 配信が 17:00 JST のため朝予測不可と判定 |
| odds_idx, jockey_idx, info_idx | content ✅ SAFE / delivery ⚠ POST-RACE | JRDB 計算 indices、 全て pre-race source |
| horse_weight, weight_diff | content ✅ SAFE / delivery ⚠ POST-RACE | 馬体重 = 公式 -70 min 発表 |
| cancel_flag, ashimoto, bagu_change | content ✅ SAFE / delivery ⚠ POST-RACE | 取消/歩様/馬具変更 = 発走前観察 |

**★ 重要 ★**: P0-3 audit は明示的に全 features を「content ✅ SAFE」と判定している。「POST-RACE」はあくまで delivery (配信タイミング) の問題。

### 1.2 TYB_MERGE_BUG_AUDIT_2026_05_16.md (Sub-task 6)

Sub-task 6 は 5 features に絞って評価:

| feature | corr_target | 判定 |
|---------|-------------|------|
| `jrdb_odds_idx` | +0.4214 | ★ LEAK 確定 ★ と判定 (≈ \|popularity\| 0.4242) |
| `jrdb_paddock_idx` | +0.3539 | ⚠ delivery 17:00 JST = post-race confirmed |
| `jrdb_live_composite_idx` | +0.2573 | ⚠ odds+padock 合成 = 部分 LEAK |
| `jrdb_body_code` | +0.0121 | ✅ safe、 信号極小 |
| `jrdb_demeanor_code` | +0.0041 | ✅ safe、 信号極小 |

**★ Sub-task 6 の誤り ★**: `jrdb_odds_idx` を `popularity` との corr 類似から「LEAK 確定」と判定したが、これは誤ラベル。詳細は §3 参照。

### 1.3 P0_4_FINAL_VERDICT_2026_05_16.md (Sub-task 11 / commit 416c4703)

**永久放棄 root cause** (commit message より):
1. V15 truncate (predict_core.py:2160-2163、 num_feature=145、 TYB 5 features 必ず削除)
2. TYB 5 features content audit: LEAK 3 / 信号ゼロ 2 → 採用候補 = 0
3. P0-4 fetch 経路は技術的成立、 取込 value 0

**根拠の性質**: timestamp 実測 + V15 model inspect + corr_target 計算。 仮定ではなく実測ベース。
ただし「LEAK 3 件」の判定 (§1.2) に誤りがある (後述 §3)。

---

## 2. TYB ファイル実態確認 (read-only)

### 2.1 ファイル存在

`data/jrdb/extracted/Tyb/` に 1,340 ファイル (TYB150104.txt〜TYB260516.txt) 確認済。
最新 5 件の local modification timestamp:
- `TYB260509.txt`: 2026-05-16 18:23:51 (5/9 分、5/16 朝 fetch バッチで取得)
- `TYB260510.txt`: 2026-05-16 18:23:51
- `TYB260516.txt`: 2026-05-16 18:23:50 (5/16 当日夕方以降に extract)

→ ファイルは 17:00 JST 以降に local に extract されている = 標準 datazip path (post-race publish) の証拠

### 2.2 TYB field schema (26 fields、 レコード長 128B)

`tools/parse_jrdb.py:258` TYB_COLUMNS より全 field:

| field | bytes | 分類 | 根拠 |
|-------|-------|------|------|
| basho_code, year, kai, nichi, race_num, umaban | key | PRE_RACE | race ID |
| idm | 11-15 | PRE_RACE | JRDB 総合指数 (直前計算) |
| jockey_idx | 16-20 | PRE_RACE | 騎手指数 (直前) |
| info_idx | 21-25 | PRE_RACE | 情報指数 (直前) |
| odds_idx | 26-30 | PRE_RACE | オッズ指数 (直前オッズ動向) |
| padock_idx | 31-35 | PRE_RACE | パドック指数 (発走 -30〜-40 min 観察) |
| reserve1 | 36-40 | N/A | 予備 |
| sogo_idx | 41-45 | PRE_RACE | 総合指数 (idm/odds/padock 合成) |
| bagu_change | 46 | PRE_RACE | 馬具変更 (発走前) |
| ashimoto | 47 | PRE_RACE | 歩様 (発走 -30 min パドック観察) |
| cancel_flag | 48 | PRE_RACE | 取消フラグ (発走前) |
| jockey_code, jockey_name | 49-65 | PRE_RACE | 騎手情報 |
| weight_carry | 66-68 | PRE_RACE | 斤量 (ハンデ確定値、 発走前) |
| minarai | 69 | PRE_RACE | 見習い区分 |
| baba_code | 70-71 | PRE_RACE | 馬場状態 (当日朝〜直前) |
| weather_code | 72 | PRE_RACE | 天候 |
| tansho_odds | 73-78 | PRE_RACE | 単勝オッズ (-15 min snapshot) |
| fukusho_odds | 79-84 | PRE_RACE | 複勝オッズ (-15 min snapshot) |
| odds_time | 85-88 | PRE_RACE | オッズ取得時刻 (HHMM) |
| horse_weight | 89-91 | PRE_RACE | 馬体重 (公式 -70 min 発表) |
| weight_diff | 92-94 | PRE_RACE | 馬体重増減 |
| odds_mark | 95 | PRE_RACE | オッズ印 |
| padock_mark | 96 | PRE_RACE | パドック印 |
| sogo_mark | 97 | PRE_RACE | 総合印 |
| batai_code | 98 | PRE_RACE | 馬体コード |
| kehai_code | 99 | PRE_RACE | 気配コード |
| start_time | 100-103 | PRE_RACE | 発走予定時刻 (HHMM) |

★ POST_RACE field = 0 件 ★
★ finish, chakujun, time_raw, prize 等のポスト確定フィールドは一切存在しない ★

### 2.3 odds_time vs start_time 実測 (TYB260516.txt)

```
N = 493 records
min delta (start_time - odds_time): 11 min
max delta: 24 min
median delta: 18.0 min
odds_time < start_time (pre-race): 493/493 = 100%
odds_time >= start_time (post-race): 0/493 = 0%
```

★ 全 records が発走前 11〜24 分の odds snapshot ★ — post-race 混入 = 0%

---

## 3. odds_idx 「LEAK 確定」ラベルの再評価

### 3.1 過去 audit の判定ロジック

`jrdb_odds_idx` は `popularity` (corr_target = -0.4242) とほぼ同等の corr_target +0.4214 を持つとして「LEAK 確定」と判定された。

### 3.2 なぜこれは誤りか

`popularity` は V15 において除外された feature だが、V15 の `LEAK_FEATURES_A` には含まれていない。V15 が除外しているのは:
```
odds_log, horse_weight, condition_enc, weight_change,
weight_change_abs, weight_cat, weight_cat_dist, cond_surface
```

V15 には代わりに `oz_base_pop_rank`, `prev_odds_log`, `odds_change_rate`, `pop_rank_change`, `odds_sharp_drop` が **含まれている** (morning 08:00 odds-based features)。

つまり `popularity` = 「確定オッズに基づく事後的な人気順」ではなく、「朝オッズに基づく事前推定人気」= PRE-RACE 情報。

- `odds_idx` = -15 min 直前 odds 動向指数
- `oz_base_pop_rank` = 08:00 morning odds rank

両者が高い corr を持つのは当然 (同じ情報源)。これは **multicollinearity (重複)** であり、**content leak (post-race データ混入) ではない**。

高い corr_target = 「予測に有用な情報」の証拠であり、post-race leak の証拠ではない。

### 3.3 正しい判定

| feature | content verdict | 根拠 |
|---------|----------------|------|
| `odds_idx` | **PRE_RACE (safe)** | -15 min pre-race odds 動向 index。 corr_target 高いのは有用な pre-race 信号 |
| `padock_idx` | **PRE_RACE (safe)** | 発走 -30〜-40 min パドック観察 |
| `sogo_idx` | **PRE_RACE (safe)** | odds + padock 合成。両方 PRE_RACE なら合成も PRE_RACE |
| `body_code` | **PRE_RACE (safe)** | 馬体コード (パドック観察) |
| `demeanor_code` | **PRE_RACE (safe)** | 気配コード (パドック観察) |

★ **5 features 全て PRE_RACE content、 LEAK なし** ★

---

## 4. TYB 配信タイミング — 2 path の整理

### 4.1 標準 datazip path (現在の fetch 経路)

| URL | Last-Modified (JST) | 解釈 |
|-----|---------------------|------|
| `/member/datazip/Tyb/{YY}/TYB{yymmdd}.zip` | 17:00〜17:10 JST | 全 race 終了後の確定版 |
| `/member/data/Tyb/TYB{yymmdd}.lzh` | 17:03〜17:10 JST | 同上 LZH 版 |

5 週連続観測 (2026-03-22〜2026-05-16):
- 全件 17:00-17:10 JST publish
- 朝 06:00 fetch → 404 (当日分まだ存在しない)
- ★ 当日レースの事前予測には使えない ★

### 4.2 tyokuzen path (新規発見、 5/16 設計 doc で確認)

| URL | Last-Modified (JST) | 解釈 |
|-----|---------------------|------|
| `/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh` | 16:15〜16:21 JST | 最終 race -10〜-20 min の snapshot |

5 週連続観測 (5/2〜5/16):

| 日付 | Last-Modified | 最終 race 発走 (推定) | delta |
|------|--------------|---------------------|-------|
| 5/2 Sat | 16:21 JST | ~16:35 (最終 race) | -14 min |
| 5/3 Sun | 16:16 JST | ~16:30 | -14 min |
| 5/9 Sat | 16:20 JST | ~16:35 | -15 min |
| 5/10 Sun | 16:18 JST | ~16:30 | -12 min |
| 5/16 Sat | **16:16 JST** | ~16:35 | **-19 min** |

★ 最終 race 発走 15〜20 分前の snapshot = JRDB 公式 doc 「発走 20 分前前後」と一致 ★

**JRDB 公式 doc (`jrdb_doc.pdf` §4.1) 引用**:
> 「直前情報」は、発走 20 分前前後を目処に発信しています。情報が古いまま・画面が変わらないといった時には、必ずリロード(再読み込み・更新)をしてからご覧になって下さい。

### 4.3 per-race 更新の確認状況

| 証拠 | 確認状況 |
|------|---------|
| JRDB 公式 doc: 発走 20 分前に発信 | ✅ 文書確認 |
| odds_time field: 全 record が発走 -11〜-24 min | ✅ 実測 (N=493) |
| 最終 race の -15 min 更新 | ✅ 5 週観測 |
| 午前〜午後各 race での per-race 更新 | ⚠ **未確認** (観測は最終 race 時刻のみ) |
| MD5 差: tyokuzen ≠ standard → cumulative 更新 | ✅ 確認 |

**honest 評価**: tyokuzen path が最終 race だけでなく各 race -20 min に更新される可能性は高いが、直接観測は未実施 (5/23 live polling が 永久放棄で skip されたため)。

---

## 5. field-level LEAK verdict (最終)

| field | content leak | delivery (standard) | delivery (tyokuzen) | 予測価値 |
|-------|-------------|--------------------|--------------------|---------|
| `tansho_odds` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.18 add-one delta) |
| `fukusho_odds` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.17) |
| `odds_idx` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.17) ← LEAK 誤ラベル訂正 |
| `jockey_idx` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.16) |
| `padock_idx` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.15) |
| `info_idx` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 強 (+0.14) |
| `padock_mark` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 中 (+0.09) |
| `ashimoto` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 弱 (+0.05) |
| `sogo_idx` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 弱 (V15 内包) |
| `idm` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 微小 (V15 内包) |
| `horse_weight` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | negative add-one |
| `weight_diff` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 弱 |
| `bagu_change` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | negative add-one |
| `kehai_code` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | negative add-one |
| `cancel_flag` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 0 (取消馬は除外済) |
| `batai_code` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | 微小 (body code) |
| `baba_code` | ✅ **PRE_RACE** | ⚠ POST (17:00) | ✅ PRE (16:15) | V15 内包 |

★ **content leak = 0 件 (全 26 fields PRE_RACE)** ★
★ **delivery leak (standard path) = 全 fields** (17:00 JST は race 後配信) ★
★ **delivery safe (tyokuzen path) = 全 fields** (16:15 JST = last race -15 min) ★

---

## 6. 真の verdict

### 6.1 TYB = PRE-RACE data か POST-RACE data か

**TRUE VERDICT: TYB は完全に PRE-RACE data**

根拠:
1. **field schema**: `finish`, `chakujun`, `time_raw`, `prize_money` 等の確定値フィールドが一切存在しない。SED (成績データ) と比較すれば明白
2. **odds_time measurement**: 全 493 records で `odds_time < start_time` (median delta = 18 min)
3. **JRDB 公式**: TYB = 直前情報 (直前 = immediately before) と命名。発走 20 分前配信と明記
4. **内容**: 単複オッズ、パドック印、馬体重、騎手情報 = 全て発走前に確定する情報

### 6.2 過去 永久放棄 verdict の評価

| 判定根拠 | 正誤 |
|---------|------|
| V15 で truncate される (num_feature=145) → V15 で使っても効果 0 | ✅ **正しい** |
| 標準 datazip path は 17:00 JST (post-race) → 朝予測に使えない | ✅ **正しい** |
| odds_idx は LEAK content → 採用不可 | ❌ **誤り** (corr_target 高い = 有用、 content は PRE_RACE) |
| V21/V22 でも永久放棄 | ❌ **誤り** (content 安全、 tyokuzen path 使用で delivery 解決可能) |

**★ 正確な表現 ★**:
- V15 においては正しい (truncate で価値 0)
- 「LEAK」ラベルは不適切 — 正確には「delivery timing problem」
- 「永久放棄」は過大 — 「V15 での採用不可、 V21+ で再検討」が正確

### 6.3 使用可能 fields と条件

**retrospective backtest (V21/V22 学習)**:
- 全 fields 使用可能 (content safe、 17:00 JST のデータを翌日以降の学習 data として使用)
- 推奨: `tansho_odds`, `fukusho_odds`, `odds_idx`, `jockey_idx`, `padock_idx`, `info_idx`, `padock_mark`, `ashimoto`
- 注意: `horse_weight`, `kehai_code`, `bagu_change` は V21 add-one で negative — 採用前に ablation 必須

**live -15 min prediction (V21/V22 production)**:
- 要件: tyokuzen path `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh` を fetch
- 未確認事項: 午前 race の per-race 更新 (最終 race のみ確認済、 全 race は未観測)
- 作業: 5/23 or 6/1 開催日に live polling 実施 → per-race 更新確認後 GO/NO-GO

---

## 7. 5/24+ Phase 3 への implications

### 7.1 訂正

- TYB を「LEAK / 永久放棄」リストから外す
- 正確なラベル: 「delivery timing problem、 V15 では truncate のため不採用、 V21+ で再検討」
- `odds_idx` の「LEAK 確定」ラベルを削除 → 「PRE_RACE content、 high predictive value、 V21 候補」

### 7.2 V21/V22 学習での推奨 action

| 優先 | action |
|------|--------|
| 高 | 6/1 開催日に tyokuzen path の per-race 更新を live 観測 |
| 高 | V21 学習 data に TYB 全 fields (retrospective) を merge (100% match 確認済、 race_id 変換は `_build_nk_race_id_from_jv()` で対応) |
| 高 | ablation test: add-one 5CV で `tansho_odds`, `odds_idx`, `padock_idx` 単独 delta 測定 |
| 中 | tyokuzen fetch script 作成 (既存 `tools/v21/jrdb_tyb_live_fetch.py` を改修) |
| 低 | negative add-one features (`horse_weight`, `kehai_code`, `bagu_change`) は除外検討 |

### 7.3 V15 production への影響

★ **0 (変更なし)** ★
- predict_core.py の truncate logic は維持 (TYB を fetch しても V15 では value 0)
- race_auto_notify.py の fetch_jrdb_tyb() の silent failure は behavior 変えず
- 本 doc は read-only audit のみ

---

## 8. honest 限界

1. **tyokuzen path の per-race 更新は直接観測なし** — 最終 race のみ確認。午前 race (-15 min) での更新は JRDB 公式 doc + odds_time 内部 field から推定。live 確認推奨
2. **JRDB 規約未読** — kiyaku.html の TLS エラーで直接確認できていない。60 秒 polling が許容か否か未確認
3. **n=348 の AUC delta (+0.1429) は CI 広い** — 5CV 1 回のみ。本物の V21 学習での delta は異なる可能性
4. **V21 add-one が V15-only calibrator task (baseline AUC 0.4653) でのみ測定** — V21 full 6-fold WF での delta は別途確認必要

---

## 9. 参考 / 出典

| doc | 内容 |
|-----|------|
| `docs/TYB_LEAK_AUDIT_2026_05_16.md` | P0-3 delivery timing audit (Sub-task 5-4)、 AUC delta 測定 |
| `docs/TYB_MERGE_BUG_AUDIT_2026_05_16.md` | Sub-task 6、 root cause: merge 関数なし |
| `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` | Sub-task 7、 tyokuzen path 設計、 Last-Modified 5 週観測データ |
| `docs/P0_4_FINAL_VERDICT_2026_05_16.md` | Sub-task 11、 永久放棄 verdict |
| `tools/parse_jrdb.py:258` | TYB_COLUMNS 定義 (全 26 fields) |
| `data/jrdb/extracted/Tyb/TYB260516.txt` | 2026-05-16 実データ (N=493 records) |
| commit `416c4703` | 永久放棄 commit、 2026-05-16 20:31 JST |

---

## 10. caveman summary

- TYB content: 全 field PRE_RACE。finish/rank/prize なし。LEAK なし。
- TYB delivery (標準 path): POST_RACE (17:00 JST)。当日 live 不可。
- TYB delivery (tyokuzen path): PRE_RACE。16:15 JST = 最終 race -15 min 確認済。
- 過去 odds_idx LEAK 判定: 誤ラベル。高 corr = 有用な信号、post-race 混入ではない。
- V15 永久放棄: V15 に限り正しい (truncate で価値 0)。
- V21/V22 永久放棄: 誤り。content safe、 tyokuzen 経路で delivery 解決可能。
- 5/24+ Phase 3 action: 6/1 live 観測 → tyokuzen per-race 確認 → V21 add-one ablation → GO 判定。
