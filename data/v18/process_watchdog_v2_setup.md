# ProcessWatchdog v2 schtasks 登録手順

**作成**: 2026-05-05 夜 / 緊急 3 件 #1
**前提**: タスクスケジューラ `ProcessWatchdog` 既存 (Disabled、v1 PID ベース)
**目標**: v2 (ログ鮮度ベース) に切替えて Enable + 静音化

## 1. 現状

```
TaskName              State LastResult
--------              ----- ----------
ProcessWatchdog    Disabled          0  ← v1 で監視対象 0、4/19 audit で実質無効と判明
```

実体ファイル:
- `tools/process_watchdog_v2.py` (290 行、ログ mtime ベース、`--once` モード対応)
- `tools/task_watchdog_v2.bat` (24 行、`--once` 1 回実行 → schtasks Trigger で 5 分間隔)
- `tools/silent_runner.vbs` (24 行、wscript hidden 起動)
- **本書 同梱**: `tools/register_process_watchdog_v2.ps1` (admin 権限で実行)

## 2. 監視対象

| name | log_glob | stale 閾値 | 再起動コマンド |
|------|----------|-----------|---------------|
| daily_predict | `daily_predict*.log` | 30 分 | `python -u tools/daily_predict.py --resume` |
| race_auto_notify | `race_auto_notify*.log` | 10 分 | `python -u tools/race_auto_notify.py` |

再起動ポリシー: 07:00-18:00 のみ自動再起動、それ以外は Discord 警告のみ
通知: `notify_done.py` 経由、`#updates` channel に `🚨 CRITICAL` prefix

## 3. 登録手順 (admin PowerShell で実行)

```powershell
# 動作確認のみ (実変更なし)
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1 -DryRun

# 本実行 (admin 権限必要)
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1
```

ps1 がやること:
1. 既存 `ProcessWatchdog` タスクが存在するか確認
2. TR を `wscript.exe silent_runner.vbs task_watchdog_v2.bat` に変更 (静音化込み)
3. トリガを表示 (5 分間隔の Repetition があるか確認)
4. タスクを Enable
5. 動作確認 (`process_watchdog_v2.py --once --dry-run`)
6. 最終状態表示

## 4. ロールバック

```powershell
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1 -Rollback
```

→ ProcessWatchdog を再度 Disable に戻す (TR 変更は維持される、必要なら schtasks /Change で元の TR に手動戻し)。

## 5. 5 分間隔の Trigger 設定

既存の Trigger が 5 分間隔でなければ、以下を admin PowerShell で実行:

```powershell
$task = Get-ScheduledTask -TaskName "ProcessWatchdog"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)
Set-ScheduledTask -TaskName "ProcessWatchdog" -Trigger $trigger
```

## 6. 動作確認

ps1 実行後:

```powershell
# 状態確認
Get-ScheduledTask -TaskName "ProcessWatchdog" | Select TaskName,State

# ログ確認 (5 分後以降)
Get-Content C:\Users\takum\keiba-ai\logs\watchdog_v2_20260505.log -Tail 20
```

5 分後にログが書かれていれば成功。

## 7. 既存実装との違い

| 項目 | v1 (process_watchdog.py) | v2 (process_watchdog_v2.py) |
|------|---|---|
| 検知方法 | PID 登録ファイル | logs/*.log の mtime |
| 監視対象登録 | ad-hoc (PID json) | ハードコード TARGETS |
| ゾンビ検知 | ❌ プロセス生きてれば OK | ✅ ログ更新止まってたら STALE |
| 再起動 | あり | あり (07:00-18:00 のみ) |
| 通知 | あり | あり (CRITICAL prefix + 色 red) |
| daily_predict_watchdog (S4) との関係 | 別系統 | 別系統 (重複監視で問題なし、daily_predict_watchdog は subprocess 監視 + Cookie auto refresh) |

v2 は 5 分おき軽量チェック、v1 は不使用。daily_predict_watchdog は別途継続。

## 8. 失敗時対応

ps1 が失敗する典型ケース:
- admin 権限なし → PowerShell を「管理者として実行」で起動
- schtasks /Change エラー → 手動で `taskschd.msc` を開いて編集
- silent_runner.vbs のパスが違う → `C:\Users\takum\keiba-ai\tools\silent_runner.vbs` 確認

それでも失敗した場合は Discord #updates に貼って 5/6 火に再対応。
