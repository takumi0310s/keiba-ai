# netkeiba スクレイパー対応表

Last updated: 2026-04-13

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
