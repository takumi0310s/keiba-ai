# タスクスケジューラ 全16件 静音化 手順書

作成日: 2026-05-04 (Phase 2.5)

## 1. 何が変わるか

ターミナル(コマンドプロンプト)ウィンドウが**画面に出てこなくなる**。
朝のちらつき、毎時 X:30 の TybPublishMonitor 出現が消える。

| 項目 | Before | After |
|------|--------|-------|
| 実行時表示 | 黒い CMD ウィンドウが点滅・常駐 | 完全非表示 |
| 動作内容 | (変わらず) | (変わらず) |
| 出力ログ | (変わらず: logs/, *.log) | (変わらず) |
| 終了コード | (変わらず) | (変わらず) |

仕組み: 各タスクの Execute を `wscript.exe` に変更し、`tools/silent_runner.vbs` が
ウィンドウ非表示で元の .bat を呼ぶ。実行内容は完全に同一。

## 2. 影響範囲

- **動作影響なし**: bat ファイルそのもの・引数・実行ユーザー・スケジュール時刻 すべて変更なし
- **対象**: 16件全て (一覧は本ファイル末尾)
- **rollback 可**: `silentify_rollback.ps1` で元に戻せる (backup JSON 保存済)

## 3. 実行手順

### 3.1 管理者 PowerShell を起動

スタートメニューで `powershell` 検索 → 右クリック → **管理者として実行**

### 3.2 リポジトリへ移動 + 実行

```powershell
cd C:\Users\takum\keiba-ai
powershell -ExecutionPolicy Bypass -File tools\silentify_all_tasks.ps1
```

期待出力:
```
[2026-05-05 ..:..:..] ===== silentify_all_tasks 開始 =====
[..] 対象タスク件数: 16
[..] [Keiba-AM3FireCheck] 変更
[..]   OLD Exec: C:\Users\takum\keiba-ai\am3_fire_check.bat
[..]   NEW Exec: wscript.exe
[..]   NEW Args: "C:\Users\takum\keiba-ai\tools\silent_runner.vbs" "C:\Users\takum\keiba-ai\am3_fire_check.bat"
[..]   -> OK
... (16件繰り返し)
[..] 成功: 16 / 16
```

ログ: `logs/silentify_YYYYMMDD_HHMMSS.log` に保存される。

## 4. 動作確認

### 4.1 1件サンプル確認

```powershell
Get-ScheduledTask -TaskName "Keiba-TybPublishMonitor" | Select-Object -ExpandProperty Actions | Format-List Execute, Arguments
```

期待:
```
Execute   : wscript.exe
Arguments : "C:\Users\takum\keiba-ai\tools\silent_runner.vbs" "C:\Users\takum\keiba-ai\tools\tyb_publish_monitor.bat"
```

### 4.2 全件一括確認

```powershell
$names = @('Keiba-AM3FireCheck','Keiba-AM6FireCheck','Keiba-AM8FireCheck','Keiba-FridayWeekendScrape','Keiba-MorningDigest','Keiba-Morning_Sat','Keiba-Morning_Sun','Keiba-NightlySanity','Keiba-PreFireCheck','Keiba-TybPublishMonitor','KeibaAI_DriftDetector','DailyJrdbKyi','DailyPredict','DailyPremiumScrape','DailyResultsEvening','DailyResults_Sat')
foreach ($n in $names) {
  $t = Get-ScheduledTask -TaskName $n
  $a = $t.Actions[0]
  "{0,-30} {1}" -f $n, $a.Execute
}
```

全て `wscript.exe` になっていれば OK。

### 4.3 実発火テスト (任意)

待ちきれない場合は手動 1件発火:
```powershell
Start-ScheduledTask -TaskName "Keiba-TybPublishMonitor"
```
ターミナルウィンドウが**出てこなければ成功**。logs/tyb_publish_monitor*.log が更新されているか確認。

### 4.4 自然発火観察

- 次の毎時 X:30 (Keiba-TybPublishMonitor) → ターミナル無発生確認
- 翌朝 03:00 / 06:00 / 08:00 / 23:00 → 観察
- 5/5 朝 03:00 DailyPremiumScrape が静音動作 → Discord scrape 完了通知到着で動作確認

## 5. rollback (元に戻す)

問題が出た場合、管理者 PowerShell で:

```powershell
cd C:\Users\takum\keiba-ai
powershell -ExecutionPolicy Bypass -File tools\silentify_rollback.ps1
```

`tools/task_silentify_backup_5_4.json` の Execute/Arguments がそのまま書き戻される。

## 6. 対象タスク 一覧 (16件)

| # | TaskName | スケジュール | 元 .bat |
|---|----------|----------|---------|
| 1 | Keiba-AM3FireCheck | 03:15 | am3_fire_check.bat |
| 2 | Keiba-AM6FireCheck | 06:15 | am6_fire_check.bat |
| 3 | Keiba-AM8FireCheck | 08:50 | am8_fire_check.bat |
| 4 | Keiba-FridayWeekendScrape | 金 10:00 | friday_weekend_scrape.bat |
| 5 | Keiba-MorningDigest | 07:00 | morning_dashboard.bat |
| 6 | Keiba-Morning_Sat | 土 06:30 | tools/morning_top_races.bat |
| 7 | Keiba-Morning_Sun | 日 06:30 | tools/morning_top_races.bat |
| 8 | Keiba-NightlySanity | 23:00 | nightly_sanity_check.bat |
| 9 | Keiba-PreFireCheck | 02:55 | pre_fire_check.bat |
| 10 | Keiba-TybPublishMonitor | 毎時 X:30 | tools/tyb_publish_monitor.bat |
| 11 | KeibaAI_DriftDetector | 週次 | drift_detector.bat |
| 12 | DailyJrdbKyi | 06:00 | tools/daily_jrdb_kyi.bat |
| 13 | DailyPredict | 08:00 | daily_predict_watchdog.bat |
| 14 | DailyPremiumScrape | 03:00 | daily_premium_scrape.bat |
| 15 | DailyResultsEvening | 20:00 | daily_results.bat |
| 16 | DailyResults_Sat | 土 18:00 | daily_results.bat |

## 7. ファイル構成

| ファイル | 役割 |
|----------|------|
| `tools/silent_runner.vbs` | wscript で hidden-window 起動するラッパー |
| `tools/silentify_all_tasks.ps1` | 16件一括変更 (admin) |
| `tools/silentify_rollback.ps1` | 巻き戻し (admin) |
| `tools/task_silentify_backup_5_4.json` | 変更前の Execute/Arguments backup |
| `data/v18/silentify_tasks_user_guide.md` | 本書 |

## 8. トラブルシューティング

| 症状 | 対処 |
|------|------|
| ERROR: 管理者権限が必要 | PowerShell を 管理者として実行 で再起動 |
| ERROR: backup JSON が見つかりません | `git pull` で最新化、または既存 backup の path を引数で指定 |
| FAILED: ScheduledTask が見つかりません | TaskName/TaskPath が変わっている可能性。`Get-ScheduledTask` で実際の TaskPath を確認 |
| 静音化したのにウィンドウが出る | wscript.exe ではなく cscript.exe で動いてないか確認。`Get-ScheduledTask -TaskName <name>` で Execute=wscript.exe か再確認 |
| .bat 内で別の cmd ウィンドウを spawn している | bat 内の `start` コマンドで明示的に新ウィンドウを開いている → bat 側の修正が必要 |

## 9. 既知の制限

- silent_runner.vbs は wscript.exe 起動を要する。Python script 等を直接 schedule している場合は対象外 (今回の16件は全て .bat 経由なので問題なし)
- bat 内の echo 出力は consoleless で破棄される (今回 全 bat が `> log_file` リダイレクト済 or notify_done.py で Discord 通知するため影響なし)
