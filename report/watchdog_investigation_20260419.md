# Process Watchdog 動作検証レポート (2026-04-19)

## TL;DR
現行の `tools/process_watchdog.py` は **`logs/pids/*.json` に登録エントリが
ゼロのまま運用されており、何も監視していなかった**。本日 AM9:05 の
daily_predict クラッシュは watchdog に検知されるすべがなかった。

## 1. 現行仕様

### 監視対象
- `logs/pids/<name>.json` に手動で `--register` したエントリのみ
- エントリ形式:
  ```json
  {"name":"bulk_scrape_upset","pid":12345,"cmd":["python","-u","tools/bulk_scrape_upset.py"],
   "cwd":"…","log":"logs/xxx.log","started_at":"ISO","restart_count":0}
  ```
- 運用タスク (daily_predict / race_auto_notify / notify_bets_all_in_one 等) は
  どれも register されていない。

### 検知条件
- `_is_pid_alive(pid)` による **PID 生存確認のみ**
  - psutil→Windows tasklist→os.kill シグナル0 の3段階フォールバック
- ログファイルの鮮度 (mtime) は一切見ていない
- したがって「ゾンビ化してるが PID は生存」などの状態は見逃す

### リカバリ動作
- PID 死亡 かつ `scraper_guard.is_scraping_allowed() == True` の場合のみ
  `subprocess.Popen(entry['cmd'])` で再起動
- 再起動不可時 (ガード時間帯) は Discord 通知のみ
- 再起動成功時も `[watchdog] restart {name}` + 色 yellow の通知のみで
  強調 (CRITICAL) ではない

## 2. 本日の動作履歴

### タスクスケジューラ
- `keiba-ai\ProcessWatchdog` を 5分間隔で登録済み
- `schtasks /Query` 結果:
  - 最終実行 2026/04/19 15:30:01 (exit code 0)
  - 次回 2026/04/19 15:35:00
  - 5分間隔で動作中

### ログ
- **`logs/watchdog*.log` / `logs/process_watchdog*.log` は存在しない**
- `logs/pids/` ディレクトリもファイルなし
- 理由: `process_watchdog.bat` が stdout/stderr をファイルにリダイレクトしない
  ため、タスクスケジューラ下で出力が消失。

## 3. AM9:05 クラッシュ検知失敗の原因

1. **監視対象未登録**
   - `logs/pids/` は空のため、cmd_once() の冒頭で
     `if not entries: print("[watchdog] no entries"); return`
     で即終了。
   - daily_predict や race_auto_notify を register する仕組みが
     手動運用前提で、自動登録フックが存在しない。

2. **scraper_guard ブロック (修正前)**
   - 土曜AM3:00 / 日曜AM9:00 などは `is_scraping_allowed()==False`
   - フェーズ1で OPERATIONAL_CALLERS ホワイトリスト導入済みだが、
     watchdog 内部の `_scraper_guard_ok()` は `is_scraping_allowed()` を
     **caller 引数なし** で呼んでいるため、依然として週末ブロックされる。

3. **ログ鮮度監視不在**
   - PID は生存してるがフリーズ (Fortran CLOSE event でプロセスは
     残るが仕事しない) など「生きてる風ゾンビ」を検知できない。

## 4. v2 設計要件 (別途 process_watchdog_v2.py で実装)

| 項目 | 現行 | v2 |
|------|------|------|
| 監視対象 | json 手動登録 | **ハードコード** (daily_predict, race_auto_notify) |
| 検知 | PID 生存のみ | **ログファイル mtime** + プロセス名存在 |
| 鮮度閾値 | - | daily_predict=30分, race_auto_notify=10分 |
| 再起動時間帯 | scraper_guard | **07:00-18:00 のみ再起動**、外は通知のみ |
| env 追加 | なし | **SCRAPER_GUARD_DISABLE=1 + KEIBA_OPERATIONAL_MODE=1** |
| Discord 色 | yellow | **red + "🚨 CRITICAL"** プレフィックス |
| 後方互換 | - | 既存ファイルは残し、`v2` で並存実装 |

## 5. 教訓

- 「登録したけど機能していない」状態が長期間続いたのは、watchdog 自身が
  「no entries」を毎5分出すだけで無警告だったから → v2 では空なら警告。
- 監視系スクリプトは**ログを必ずファイルに残す** (bat リダイレクトを強制)。
- ホワイトリスト導入 (フェーズ1) は watchdog 内の is_scraping_allowed 呼出
  にも **caller 引数を渡す** ことで初めて機能する。
- プロセス生存確認だけでは Fortran ゾンビを検知できない
  → **ログ mtime** 併用が必須。
