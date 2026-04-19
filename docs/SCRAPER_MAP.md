# netkeiba スクレイパー対応表

Last updated: 2026-04-19

## スクリプト × 出力CSV 対応表

| スクリプト | 出力CSV | 年範囲 | ソースURL | Premium Cookie | 備考 |
|-----------|---------|--------|----------|:---:|------|
| `tools/bulk_scrape_upset.py` | `netkeiba_upset_level.csv` | 2020-2026 (`--year`) | race.netkeiba/race/shutuba.html | ✓ | 波乱度Lv1-5・上位人気信頼度 |
| `tools/scrape_master_index.py` | `netkeiba_master_index.csv`, `netkeiba_track_bias.csv`, `netkeiba_race_lap.csv` | 2020-2026 (`--all_years`) | db.netkeiba/race/ | ✓ | 3分解指数＋馬場指数＋ラップ（単数） |
| `tools/scrape_super_premium.py` | `netkeiba_ai_position.csv`, `netkeiba_track_index.csv`, `netkeiba_training_eval.csv`, `netkeiba_ai_opinion.csv`, `netkeiba_ana_best.csv` | 2020-2026 (`--year`) | race.netkeiba/newspaper.html, /result.html | ✓ | 2020年でも動作確認済み |
| `tools/scrape_master_course.py` | `netkeiba_individual_lap.csv`, `netkeiba_race_laps.csv`, `netkeiba_pace_prediction.csv`, + 共有CSV | 2020-2026 (`--source all --year`) | race.netkeiba 各種 | ✓ | `--source race_laps` で複数形ラップ |
| `tools/bulk_scrape_history.py` | `netkeiba_speed_index.csv`, `netkeiba_training_times.csv` | 2020-2025 | race.netkeiba/race/speed.html | ✓ | タイム指数・調教タイム |
| `tools/bulk_scrape_comments.py` | `netkeiba_stable_comments.csv` | 2020-2024 | race.netkeiba/race/comment.html | ✓ | クラス43+のみ |
| `tools/scrape_premium_data.py` | `netkeiba_training_times.csv`, `netkeiba_stable_comments.csv`, `netkeiba_race_tendency.csv` | `--year` | race.netkeiba/oikiri.html, /comment.html, /data_list.html | ✓ | 追切・コメント・傾向 |
| `tools/daily_premium_scrape.py` | `weekly_premium_cache/{date}/premium_cache.json` | 当日+2日 | race.netkeiba race_list_sub + JRDB | ✓ | 運用用事前取得（AM3:00自動） |
| `tools/scrape_weekend_thisweek.py` | `*_thisweek.csv` + master追記 | 今週のみ | race.netkeiba newspaper/shutuba/data_list | ✓ | 週末運用用 |
| `tools/weekly_premium_update.py` | `weekly_premium_cache/` | 当日のみ | race.netkeiba race_list_sub | ✓ | キャッシュ更新 |

## データ × 年別カバレッジ（2026-04-13時点）

| 出力CSV | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 全行 | 目標 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|
| `netkeiba_upset_level.csv` | 6914 | 1737 | 3950 | 5116 | 1145 | **0** ✗ | 72 | 18934 | 全年完全 |
| `netkeiba_track_bias.csv` | 3457 | 3454 | 1943 | **0** ✗ | **0** ✗ | 581 | — | 9435 | 全年完全 |
| `netkeiba_race_lap.csv` | 3332 | 3327 | 1872 | **0** ✗ | **0** ✗ | 565 | — | 9096 | 全年完全 |
| `netkeiba_race_laps.csv` | **0** ✗ | **0** ✗ | **0** ✗ | **0** ✗ | **0** ✗ | 1 | — | 1 | 全年完全 |
| `netkeiba_training_eval.csv` | **0** ✗ | **0** ✗ | **0** ✗ | **0** ✗ | 47181 | 47884 | — | 95065 | 全年完全 |
| `netkeiba_master_index.csv` | ? | ? | ? | ? | ? | ? | — | 81 (壊) | 全年完全（要再構築） |

※ `netkeiba_master_index.csv` はヘッダ不整合で読み込みエラー（line 83 expected 7 saw 9）→ 再構築必要。

## 欠落データ補填手順

1. **Upset 2025**: `python -u tools/bulk_scrape_upset.py` （resume進行中14192/20733）
2. **Training Eval 2020-2023**: `python tools/scrape_super_premium.py --year 2020/2021/2022/2023 --source newspaper`
3. **Track Bias / Race Lap 2023-2024**: `python tools/scrape_master_index.py --year 2023` および `--year 2024`
4. **Race Laps(複数) 全年**: `python tools/scrape_master_course.py --source race_laps --year 2020-2026`
5. **Master Index 再構築**: 破損CSVを削除してから再実行（ヘッダ不整合のため）

推奨実行順: 5→1→3→2→4（軽いものから。IPバンリスク分散のため直列実行）

## 運用モード対応スクリプト一覧 (2026-04-19 追加)

`tools/scraper_guard.py` は金22時〜月6時の「週末ガード」を持つが、レース運用中に
動かすべきタスクは以下の `OPERATIONAL_CALLERS` ホワイトリストで**バイパス**する。
個別呼び出し側で `caller="..."` を指定するか、環境変数
`KEIBA_OPERATIONAL_MODE=1` でプロセス単位で全バイパス可能。

### ホワイトリスト対応スクリプト

| caller 識別子 | スクリプト | 週末ガード挙動 | 備考 |
|---------------|----------|---------------|------|
| `daily_predict` | `tools/daily_predict.py` | **常に許可** | --resume 対応、Fortran対策済 |
| `race_auto_notify` | `tools/race_auto_notify.py` | **常に許可** | Discord 5分前通知 |
| `notify_bets_all_in_one` | `tools/notify_bets_all_in_one.py` | **常に許可** | 朝の一括通知 |
| `jrdb_health_check` | `tools/jrdb_health_check.py` | **常に許可** | 土日 AM7:30 |
| `daily_jrdb_kyi` | `tools/scrape_jrdb.py`経由 | **常に許可** | AM6:00 JRDB取得 |
| `daily_premium_scrape` | `tools/daily_premium_scrape.py` | **土日 03:00-05:59 のみ許可** | それ以外は `mode="exit"` で即終了 |

### ガード対象（従来通り停止）

以下の caller は指定しない、あるいは `OPERATIONAL_CALLERS` 外の caller なので
週末は `sys.exit(exit_code)` または 600秒 sleep ループで停止する。

- `bulk_scrape_upset`
- `bulk_scrape_history`
- `bulk_scrape_comments`
- `scrape_speed_index`
- `scrape_shinba_eval`
- `scrape_race_review`
- `scrape_premium_data`
- `scrape_newspaper_ai`
- `scrape_master_index`
- `scrape_master_course`
- `scrape_super_premium`
- `scrape_training_bulk` / `scrape_training_bulk_2025`
- `scrape_comments_bulk`
- `scrape_stable_comment`
- `scrape_weekend_thisweek`
- `scrape_data_analysis`
- `scrape_missing_all` (手動チェックのみ)

### デフォルト挙動と切替え

```python
# 従来通り (引数なし) — 運用外タスクと同じ扱い
from tools.scraper_guard import check_scraping_allowed
check_scraping_allowed()               # 週末は 600秒 sleep ループ (wait mode)
check_scraping_allowed(mode="exit")    # 週末は sys.exit(0)

# 運用タスク — ガード無視
check_scraping_allowed(caller="daily_predict")
check_scraping_allowed(caller="race_auto_notify", mode="exit")

# env 指定 (プロセス単位で全タスクバイパス)
#   KEIBA_OPERATIONAL_MODE=1 python tools/daily_predict.py
#   task_daily_predict.bat は既定で設定済
```

### `KEIBA_OPERATIONAL_MODE=1` の使い方

タスクスケジューラの bat 冒頭で `set KEIBA_OPERATIONAL_MODE=1` を
設定すると、そのプロセス内で呼び出す全スクレイパーがガードを無視する。
Fortran/OMP 対策や `SCRAPER_GUARD_DISABLE=1` も併せて設定するのが望ましい:

```bat
@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
set SCRAPER_GUARD_DISABLE=1
set KEIBA_OPERATIONAL_MODE=1
python -u tools\daily_predict.py --date %TARGET_DATE% --resume
```

### 追加時のチェックリスト

新しい運用必須スクリプトを追加するとき:

1. `tools/scraper_guard.py` の `OPERATIONAL_CALLERS` に caller 識別子を追加
2. 呼び出し側で `check_scraping_allowed(caller="<識別子>")` を明示
3. `tests/test_scraper_guard.py` のパラメータに caller を追加 (`test_operational_callers_always_allowed`)
4. この SCRAPER_MAP.md のテーブルに行追加
5. `docs/incident_report_*.md` で当該スクリプトが事故当日に関係していたら記録
