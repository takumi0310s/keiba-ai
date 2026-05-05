# NAR schtasks 登録 手順書 (admin 必要)

作成日: 2026-05-05 (Phase 2.5+)

---

## 1. 何をするか

NAR daily 自動 pipeline 用の Windows schtasks 5 件を登録する。
全タスクは `tools/silent_runner.vbs` 経由で hidden 実行 (静音化済)。

| TaskName | スケジュール | 用途 |
|----------|--------------|------|
| Keiba-NarMidDayCalendar | DAILY 13:00 | NAR 当日カレンダー (placeholder) |
| Keiba-NarDailyScrape | DAILY 16:30 | NAR 当日出馬表 + 前夜オッズ (placeholder) |
| **Keiba-NarDailyPredict** | DAILY 17:00 | **NAR 推論 + 候補抽出 (実装済)** |
| Keiba-NarLiveOddsRefresh | DAILY 19:00 | NAR live odds (placeholder) |
| Keiba-NarDailyResults | DAILY 21:30 | NAR 結果照合 (placeholder) |

placeholder は同じ `tools/nar_daily_pipeline.bat` を呼ぶ (内容は no-op 相当)。
将来 個別 script 完成後、`Set-ScheduledTask` で Action だけ書き換え。

## 2. 既存 task と時刻衝突

| 時刻 | 既存 (JRA系) | 新規 (NAR系) |
|------|--------------|---------------|
| 03:00 | DailyPremiumScrape | - |
| 06:00 | DailyJrdbKyi | - |
| 06:30 | Keiba-Morning_Sat/Sun | - |
| 07:00 | Keiba-MorningDigest | - |
| 08:00 | DailyPredict | - |
| 13:00 | - | **NarMidDayCalendar** |
| 16:30 | - | **NarDailyScrape** |
| 17:00 | - | **NarDailyPredict** |
| 19:00 | - | **NarLiveOddsRefresh** |
| 20:00 | DailyResultsEvening | - |
| 21:30 | - | **NarDailyResults** |
| 23:00 | NightlySanity | - |

→ **時刻衝突なし**。

## 3. 実行手順

### 3.1 admin PowerShell 起動

スタートメニュー → `powershell` 検索 → 右クリック → **管理者として実行** → UAC OK

### 3.2 リポジトリで実行

```powershell
cd C:\Users\takum\keiba-ai
powershell -ExecutionPolicy Bypass -File tools\register_nar_schtasks.ps1
```

期待出力:
```
[..] ===== register_nar_schtasks 開始 =====
[..] [Keiba-NarMidDayCalendar] 登録
[..]   -> OK
... (5 件)
[..] 成功: 5 / 5
```

ログ: `logs/register_nar_schtasks_YYYYMMDD_HHMMSS.log`

## 4. 登録確認

```powershell
Get-ScheduledTask -TaskName "Keiba-Nar*" | Format-Table TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}} -AutoSize
```

期待:
```
TaskName                  State NextRun
--------                  ----- -------
Keiba-NarMidDayCalendar   Ready 2026-05-06 13:00:00
Keiba-NarDailyScrape      Ready 2026-05-06 16:30:00
Keiba-NarDailyPredict     Ready 2026-05-06 17:00:00
Keiba-NarLiveOddsRefresh  Ready 2026-05-06 19:00:00
Keiba-NarDailyResults     Ready 2026-05-06 21:30:00
```

Action 詳細確認:

```powershell
(Get-ScheduledTask -TaskName "Keiba-NarDailyPredict").Actions | Format-List Execute, Arguments, WorkingDirectory
```

期待:
```
Execute          : wscript.exe
Arguments        : "C:\Users\takum\keiba-ai\tools\silent_runner.vbs" "C:\Users\takum\keiba-ai\tools\nar_daily_pipeline.bat"
WorkingDirectory : C:\Users\takum\keiba-ai
```

## 5. 動作確認 (手動 1 回発火)

```powershell
Start-ScheduledTask -TaskName "Keiba-NarDailyPredict"
```

ターミナルウィンドウが**出ない**ことを確認。
ログ: `logs/nar_daily_YYYYMMDD.log` が更新されれば成功。

## 6. 削除/再登録

設計変更時は再実行で OK。`register_nar_schtasks.ps1` は既存タスク削除→新規作成 を行う。

完全削除のみ:

```powershell
Get-ScheduledTask -TaskName "Keiba-Nar*" | Unregister-ScheduledTask -Confirm:$false
```

## 7. トラブルシューティング

| 症状 | 対処 |
|------|------|
| ERROR: 管理者権限 | PowerShell admin で再起動 |
| nar_daily_pipeline.bat not found | git pull で latest 取得 |
| FAILED Register-ScheduledTask: ... | task name 衝突なら自動削除→再作成 (script 内対応済) |
| 動作後 logs/nar_daily_*.log が空 | predict_nar.py 実行失敗。手動 `python tools\predict_nar.py --date 20260505` でデバッグ |
| Discord 通知来ない | `tools/notify_done.py` 単独動作テスト |

## 8. 5/12 (火) 開始までの前提

- [x] tools/predict_nar.py 汎用版 完成 (Phase 2.5+ session)
- [ ] tools/scrape_nar_today.py 実装 (placeholder 解消)
- [ ] tools/scrape_nar_results.py 実装
- [x] silent_runner.vbs 静音化 適用済
- [ ] **本手順書による schtasks 登録** ← user 作業

完了後、5/12 (火) 17:00 に NAR 推論が自動発火、Discord 通知到着で動作確認。
