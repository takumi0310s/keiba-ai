# Phase 26 scraper 実装報告 (Task #10 + #12)

**日付**: 2026-05-12
**branch**: `worktree-agent-a723b775ade67456d`
**実装者**: agent (Opus 4.7)

---

## 1. 概要

QA Audit (5/11) で確認した 2026 完全未取得 csv 群の補完 wrapper、
および JRDB 火曜先行 type (KTA / MZA / MSA) の standalone scraper を新規実装。

V15 投資保護 (predict_core / daily_predict / model 不変、 push 禁止、
SCRAPER-GUARD 範囲) を厳守。 既存 file への破壊的変更は無し。

---

## 2. 成果物

| ファイル | 役割 | LOC |
|---|---|---|
| `tools/netkeiba_2026_catchup.py` | netkeiba 2026 csv 補完 wrapper | 約 310 |
| `tools/scrape_jrdb_kta_mza.py` | JRDB KTA/MZA/MSA scraper | 約 410 |
| `data/v18/phase26_scraper_5_12.md` | 本実装報告 (本 file) | - |

両 file ともに `py_compile` PASS。

---

## 3. Task #10: netkeiba 2026 catchup

### 3-1. 実装方針

- 新規 scraper logic を書かず、 既存 `scrape_*.py` を **subprocess.run** で連続実行する wrapper。
- 各 sub-scraper は自前で resume / dedup ロジックを持つため、 wrapper は
  job 単位の `argv` 構築 + 進捗集計のみに専念。
- `--dry-run` で 各 csv の 2026 不足件数を表示 (取得は行わない)。
- `--types` で job を絞り込み可能 (master / review / super_premium /
  shinba / training_times / upset)。
- `--limit` 指定で 各 scraper に 1 回あたりの上限件数を委譲。

### 3-2. job 一覧 (sub-scraper 単位)

| job key | sub-scraper | 補完 csv |
|---|---|---|
| `master` | `scrape_master_index.py` | netkeiba_master_index.csv / netkeiba_track_bias.csv / netkeiba_race_lap.csv |
| `review` | `scrape_race_review.py` | netkeiba_race_review.csv |
| `super_premium` | `scrape_super_premium.py` | netkeiba_ai_position / ai_opinion / ana_best / track_index / training_eval (5 csv) |
| `shinba` | `scrape_shinba_eval.py` | netkeiba_shinba_eval.csv |
| `training_times` | `scrape_premium_data.py` | netkeiba_training_times.csv |
| `upset` | `bulk_scrape_upset.py` | netkeiba_upset_level.csv |

合計 6 jobs / 12 csv を 1 コマンドでカバー。

### 3-3. dry-run 結果 (2026 年、 jra_races_full.csv ベース)

```
[dry-run] year=2026  total target races: 385
  master       : missing_total= 1155 (3 csv x 385)
  review       : missing_total=  385
  super_premium: missing_total= 1925 (5 csv x 385)
  shinba       : missing_total=  385
  training_times: missing_total= 385
  upset        : missing_total=  245 (140 already done)
```

`netkeiba_upset_level.csv` のみ 140 R 取得済 (5/3 まで)。
それ以外の 11 csv は 2026 完全 0 R から開始。

### 3-4. 使用例

```bash
# 件数確認のみ
python tools/netkeiba_2026_catchup.py --dry-run

# 全 csv 補完 (順次実行)
python tools/netkeiba_2026_catchup.py

# 1 csv 群だけ実行
python tools/netkeiba_2026_catchup.py --types super_premium

# テスト: 各 scraper 50R 上限
python tools/netkeiba_2026_catchup.py --limit 50
```

### 3-5. 安全策

- SCRAPER-GUARD (Fri22:00-Mon06:00) は wrapper 起動時に check して即終了。
  各 sub-scraper も同 guard を内包 (`from tools.scraper_guard import check_scraping_allowed`)。
- predict_core / daily_predict / モデル file には一切触れない。
- subprocess 経由なので、 既存 scraper のロジックに副作用なし。

---

## 4. Task #12: JRDB KTA / MZA / MSA scraper

### 4-1. 実装方針

- `tools/scrape_jrdb.py` の record-byte / `download_parse_jrdb_batch2.py` の
  parse_kta_line を踏襲、 standalone な `--date` / `--range` / `--weeks`
  対応 scraper を新規実装。
- JRDB は **火曜更新** なので、 dates は 自動で **火曜のみ** に絞る。
- LZH 解凍は 7-Zip 優先、 `lhafile` fallback。
- credentials は `.env` の `JRDB_ID` / `JRDB_PASSWORD` を使用 (既存と同じ)。
- 出力先:
  - `data/jrdb_kta.csv` (既存 64 MB に append + dedup [race_id, blood_num])
  - `data/jrdb_mza.csv` (新規; dedup [blood_num, delete_date])
  - `data/jrdb_msa.csv` (新規; dedup [blood_num, delete_date])

### 4-2. KTA parser

既存 `download_parse_jrdb_batch2.py:parse_kta_line` と同等の 386-byte schema:

| field | byte | 説明 |
|---|---|---|
| race_id | 1-8 | 場+年+回+日+R から生成 |
| blood_num | 49-56 | 血統登録番号 |
| horse_name | 57-92 | 競走馬名 |
| jockey_name | 97-108 | 騎手名 |
| weight_carry | 109-111 | 斤量 (0.1kg) |
| idm | 129-133 | IDM |
| ten_idx_pred | 342-346 | テン指数予想 |
| pace_idx_pred | 347-351 | ペース指数予想 |
| agari_idx_pred | 352-356 | 上がり指数予想 |
| ichi_idx_pred | 357-361 | 位置指数予想 |
| (約 30 column) | | |

→ 既存 jrdb_kta.csv schema と互換 (append 可能)。

### 4-3. MZA / MSA parser

JRDB 抹消馬データ 公開フォーマット仕様より、 record≧50 byte:

| field | byte | 説明 |
|---|---|---|
| blood_num | 1-8 | 血統登録番号 |
| horse_name | 9-44 | 競走馬名 |
| delete_date | 45-52 | 抹消年月日 YYYYMMDD |
| delete_reason | 53-54 | 抹消事由コード (任意) |

MZA / MSA は同 layout。 MSA は週次差分、 MZA は全 snapshot。

### 4-4. dry-run 結果 (直近 4 週)

```
JRDB KTA / MZA / MSA scraper
  types: KTA, MZA, MSA
  dates: 5 dates (2026-04-14 ... 2026-05-12)

  expected URLs (15 件):
    http://www.jrdb.com/member/data/Kta/KTA260414.lzh
    http://www.jrdb.com/member/data/Mza/MZA260414.lzh
    http://www.jrdb.com/member/data/Msa/MSA260414.lzh
    ... (各 火曜 x 3 type)
    http://www.jrdb.com/member/data/Kta/KTA260512.lzh

  outputs:
    data/jrdb_kta.csv  exists=True  size=64,068,725 B
    data/jrdb_mza.csv  exists=False
    data/jrdb_msa.csv  exists=False
```

### 4-5. 使用例

```bash
# dry-run (URL list 表示のみ)
python tools/scrape_jrdb_kta_mza.py --dry-run

# 直近 4 週分 全 type
python tools/scrape_jrdb_kta_mza.py

# 単日
python tools/scrape_jrdb_kta_mza.py --date 20260512

# 期間
python tools/scrape_jrdb_kta_mza.py --range 20260101 20260512

# KTA のみ
python tools/scrape_jrdb_kta_mza.py --types KTA
```

### 4-6. 火曜 schtask 登録 sample (将来)

```
タスク名 : JRDB-KtaMzaMsa-Weekly
トリガー : 毎週 火曜 20:30
アクション: python tools/scrape_jrdb_kta_mza.py --weeks 1
```

---

## 5. テスト結果

| 項目 | 結果 |
|---|---|
| `py_compile tools/netkeiba_2026_catchup.py` | PASS |
| `py_compile tools/scrape_jrdb_kta_mza.py` | PASS |
| `tools/netkeiba_2026_catchup.py --list-jobs` | OK (6 job 表示) |
| `tools/netkeiba_2026_catchup.py --dry-run` | OK (385 R × 12 csv = 4480 R 不足を表示) |
| `tools/scrape_jrdb_kta_mza.py --dry-run` | OK (15 URL + 出力 path 表示) |
| `tools/scrape_jrdb_kta_mza.py --range 20260501 20260512 --types KTA --dry-run` | OK (火曜 2 dates) |
| 既存 file への破壊的変更 | 無し |
| `predict_core.py / daily_predict.py` 変更 | 無し |
| push | 行っていない (worktree commit のみ予定) |

---

## 6. 残課題 / 次 step

- 実取得は 火曜 20:30+ 以降の手動実行 もしくは schtask 登録で。
- MZA / MSA の record layout は公開仕様ベースなので、 初回取得後に実データで
  byte offset を再検証 (削除事由コードが 53-54 にあるかは要確認)。
- netkeiba 2026 catchup の super_premium / training_times は 約 385 R × 数秒
  / R で 30 分〜数時間想定。 SCRAPER-GUARD 範囲 (平日朝/夜) で実行。

---

## 7. 結論

- **Task #10**: ✓ `tools/netkeiba_2026_catchup.py` 完成、 dry-run で
  2026 不足 4,480 race 件数を一覧表示。 既存 6 scraper を統合 wrapper。
- **Task #12**: ✓ `tools/scrape_jrdb_kta_mza.py` 完成、 KTA append +
  MZA / MSA 新規 csv に対応。 火曜のみ自動絞り込み。
- V15 投資保護: ✓ predict_core / daily_predict / モデル不変、 既存 file
  への破壊的変更なし、 push 禁止厳守。
