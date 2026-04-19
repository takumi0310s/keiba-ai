# 2026-04-19 インシデントレポート — 午前レース予測ロス事故

## TL;DR
日曜朝の自動予測パイプラインが AM3:00 / AM8:00 のいずれも起動せず、
福島1R (09:45) から中山7R (13:20) まで **午前〜昼過ぎのレース全てが
予測・通知なし** で発走した。AM12:48 に手動救出を開始、PM13:22 に予測CSV
再生成完了。PM以降の15レースは通常通り Discord 通知。

事故当日のうちに 5フェーズで根本修正を実施（commit `19c1185a`..`642de657`）。

## 1. 発生事象タイムライン (JST)

| 時刻 | 事象 |
|------|------|
| 03:00 | タスクスケジューラ `DailyPremiumScrape` 起動 → SCRAPER-GUARD の `wait` モードで 600秒スリープループに突入。次回起動を妨げる状態に。 |
| 06:00 | タスク `DailyJrdbKyi` 起動 → **失敗** (bat が LF 改行 + `wmic` 依存で Win11 24H2 非対応。後から判明) |
| 07:30 | タスク `JrdbHealthCheck_Sun` 起動 → 関連前処理失敗により健全性なし |
| 08:00 | タスク `DailyPredict` 起動 → 予測処理の途中 (阪神1R処理中) で `forrtl: error (200)` クラッシュ。原因は Intel Fortran (LightGBM 経由) の Windows console CLOSE event ハンドラ。CSV は 0byte で終了。 |
| 08:45 | タスク `RaceAutoNotify_Sun` 起動 → 予測CSV なしで読み込み失敗、一括通知 0 messages、以降アイドル |
| 09:45 | **福島1R 発走** — 予測・通知なし |
| 10:00〜13:20 | 阪神1R〜中山7R 計 13レース 予測・通知なし |
| 12:39 | ユーザーが状況確認のためログ確認指示。`daily_predict_manual` ログで 09:05 のクラッシュ発覚 |
| 12:48 | 手動で `python tools/daily_predict.py --date 20260419` 再実行開始 (初回プロセス) |
| 12:50 | さらに Claude がもう1本の daily_predict を起動（意図せず重複、後に安全確認済） |
| 13:22 | 初回プロセス完走、`data/daily_predictions/20260419.csv` を 35レース分で生成。投資額 24,500円。 |
| 13:25 | `race_auto_notify.py` 手動再起動。Active timers 15 (残り午後レース)、一括通知 2 + 整形通知 8 messages 送信 |
| 13:30〜 | 福島8R 以降の通知が正常運用復帰 |
| 16:30 | 最終レース (中山12R) 発走、race_auto_notify 自然終了 |

## 2. 根本原因

### 2.1 SCRAPER-GUARD の無差別適用
- `tools/scraper_guard.py` が **金曜22:00〜月曜06:00 は全スクレイパー停止** というルール
- しかし「週末こそ動かすべき運用タスク」(DailyPredict / RaceAutoNotify /
  DailyPremiumScrape のAM3:00早朝スロット) も停止対象
- `check_scraping_allowed()` の `wait` モードが 600秒sleepループで居座り、
  タスクスケジューラの次回起動を妨げる副作用もあった（4/13, 4/18 で観測済）

### 2.2 daily_predict の Windows 強制終了
- Intel Fortran ランタイムが `forrtl error (200)` で Windows console CLOSE を
  キャッチして即終了するバグを踏んだ
- `PYTHONUNBUFFERED=1` だけでは防げない
- 全レース処理後に初めて CSV を書き出す設計のため、途中クラッシュで
  `data/daily_predictions/*.csv` が **一つも残らない**

### 2.3 process_watchdog の無効状態
- `logs/pids/*.json` に監視エントリが **ゼロ**
- ProcessWatchdog タスクは 5分おきに起動するも `no entries` で即終了
- ログが bat に未リダイレクトで出力消失、「動いていない」ことに気付けない
- PID 生存のみチェックで、ログ mtime 鮮度は見ていなかった

### 2.4 JRDB SED/TYB/CYB 結合ロジック未整備 (派生課題)
- `merge_jrdb_predict_features` に SED 前走特徴量の結合ロジック自体が欠落
- 両方のパーサーを使い分けるうちに CSV が 2 行だけに破損
- モデルには PREV 系 8特徴量が常時デフォルト値で渡っていた

## 3. 影響範囲

| 対象 | 状態 |
|------|------|
| 午前レース 20 件 (09:45〜13:20) | **予測なし / 通知なし / 購入機会ロス** |
| 午後レース 15 件 (13:30〜16:30) | 手動救出後、通常通り通知 |
| 予測CSV `data/daily_predictions/20260419.csv` | 13:22 時点で 35 レース (障害1 除外) 生成済 |
| 累計ROI | 324R / ROI 120.2% / +45,920円 (4/18 までの実績) |
| データ損失 | なし (bakup 多数確保、モデル無変更) |

## 4. 実施した修正

### フェーズ1 — SCRAPER-GUARD 運用タスクホワイトリスト対応 (commit `e173f40d`)
- `OPERATIONAL_CALLERS` 導入 (daily_predict / race_auto_notify /
  notify_bets_all_in_one / jrdb_health_check / daily_jrdb_kyi)
- `check_scraping_allowed(caller=..., mode=..., exit_code=...)` に拡張
- `KEIBA_OPERATIONAL_MODE=1` env で全バイパス
- `daily_premium_scrape` 特例: 土日 03:00-05:59 のみ許可
- `daily_premium_scrape.py` を `mode="exit"` + caller 指定に修正
- `tests/test_scraper_guard.py` 新規 50 ケース

### フェーズ2 — daily_predict クラッシュ対策 (commit `1e208b97`)
- import 前に env: `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` /
  `OMP_NUM_THREADS=4` / `KMP_DUPLICATE_LIB_OK=TRUE`
- `SIGINT` / `SIGBREAK` ハンドラで graceful exit
- 各レース完了毎の **逐次CSV append + flush + fsync** (クラッシュ耐性)
- `--resume` オプション (既存CSVの済み race_id をスキップ)
- `tools/run_daily_predict.bat` / `tools/task_daily_predict.bat` 新規
  (既存 daily_predict.bat は無変更)

### フェーズ3 — process_watchdog v2 (commit `17e3f044`)
- `tools/process_watchdog_v2.py` 新規 (既存 v1 と並存)
- ハードコード監視対象: daily_predict / race_auto_notify
- 検知条件: **ログファイル mtime + プロセス存在** の2軸
  (daily_predict=30分、race_auto_notify=10分で STALE 判定)
- 再起動は 07:00-18:00 のみ、env 付与 (SCRAPER_GUARD_DISABLE=1 +
  KEIBA_OPERATIONAL_MODE=1 + Fortran/OMP対策)
- Discord "🚨 CRITICAL" 強通知
- `report/watchdog_investigation_20260419.md` に調査結果記録
- `tests/test_process_watchdog.py` 新規 18 ケース

### フェーズ4 — JRDB カラム英語統一 (commit `642de657`)
- `tools/jrdb_column_mapping.py` (Single Source of Truth)
- `tools/build_jrdb_v2_csv.py` でキャッシュlzh再パース → v2 CSV 生成
- `data/jrdb_{sed,tyb,cyb}_v2.csv` 新規 (既存CSVは無変更)
- `jrdb_features.py` `_resolve_jrdb_csv()` で v2 優先、なければ legacy
- `docs/JRDB_COLUMN_MAPPING.md` 対応表
- `tests/test_jrdb_column_mapping.py` 新規 15 ケース

### フェーズ5 — 直前の修正 (commit `19c1185a` — 前夜メンテで実施)
- SED/TYB/CYB csv を `jrdb_raw` 再パースで復元 (2行 → 10万行)
- `merge_jrdb_predict_features` に SED 結合ロジック追加
- PREV 結合率 0% → 45%+ に改善

## 5. テスト結果

| テスト | 件数 | 結果 |
|--------|------|------|
| tests/regression_test.py | 16 | PASS |
| tests/test_scraper_guard.py | 50 | PASS |
| tests/test_process_watchdog.py | 18 | PASS |
| tests/test_jrdb_column_mapping.py | 15 | PASS |
| **合計** | **99** | **ALL PASS** |

## 6. 再発防止策

1. **タスクスケジューラの bat は必ず CRLF + 絶対パス + ログリダイレクト**
   - 既に全 .bat を CRLF 正規化済 (commit `7f51d1be` で実施)
   - `task_*.bat` シリーズは全て `>> logs\xxx.log 2>&1` でログ保存
2. **運用タスクは caller ホワイトリスト必須**
   - 新規運用スクリプトを追加する際は `tools/scraper_guard.py` の
     `OPERATIONAL_CALLERS` に登録する
   - `tests/test_scraper_guard.py` のパラメタライズに含めてCIで強制
3. **クラッシュ前提の逐次保存設計**
   - daily_predict だけでなく race_auto_notify, jrdb スクレイパー系も
     「中間状態を即 flush」方針を推奨（今回は daily_predict のみ適用）
4. **watchdog は mtime + プロセス存在の2軸**
   - process_watchdog v2 (ログ鮮度) をタスクスケジューラ切替え予定
5. **ブランチ切り分け**
   - 運用稼働中のファイル (race_auto_notify, predictions CSV, jrdb_*.csv)
     を直接編集せず、`_v2` サフィックス + フォールバック読込で後方互換
6. **インシデント後は必ず pytest 全実行 + ドキュメント更新**

## 7. 未対応 / フォロー事項

- `ProcessWatchdog` タスクは v1 → v2 への手動切替を保留 (既存稼働を優先)
- `JrdbHealthCheck_Sun` の失敗原因特定 (AM7:30 ジョブ) は未完、要別途調査
- v2 CSV 運用への完全移行は段階的に。当面は legacy + v2 並存
- 本日の午前ロス分の収支影響は 20レース × 700円 = 14,000円 の機会損失
  (購入しなかったため直接損失ではなく、本来の期待利益分)
