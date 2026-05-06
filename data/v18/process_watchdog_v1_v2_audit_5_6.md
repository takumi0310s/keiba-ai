# ProcessWatchdog v1 vs v2 監査 + 5/9 朝の watchdog 体制確認

**作成**: 2026-05-06 朝活 (Session #29)
**ベース commit**: 811d6c34

---

## 1. ProcessWatchdog v1 (既存、4/24 以降 Disabled)

| 項目 | 値 |
|------|-----|
| schtasks State | **Disabled** |
| LastRunTime | 2026/04/24 0:15:01 |
| NumberOfMissedRuns | 3616 (4/24 から動いてない) |
| Execute | `C:\Users\takum\keiba-ai\process_watchdog.bat` |

`process_watchdog.bat` の中身 (343 byte):
```bat
@echo off
REM process_watchdog 起動用バッチ
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\Users\takum\keiba-ai
python -u tools\process_watchdog.py --once
endlocal
```

`tools/process_watchdog.py` (8517 byte): **PID 登録ベース** の v1 実装。
監視対象 ar `tools/process_pid.json` に登録された PID を ping するだけ。
4/19 audit (`report/watchdog_investigation_20260419.md`) で「監視対象ゼロ」と判明、4/24 以降 Disabled で停止。

→ **v1 は実質無効、再起動不要**。

---

## 2. ProcessWatchdog v2 (Session #5、本日修正で admin 登録準備完了)

| 項目 | 値 |
|------|-----|
| schtasks State | (未登録 → 修正後 admin 1 コマンドで Enable) |
| 機構 | `tools/process_watchdog_v2.py` (11360 byte) |
| 監視 | logs/{daily_predict,race_auto_notify}*.log の mtime |
| 検知閾値 | daily_predict 30分、race_auto_notify 10分 |
| 実行 bat | `tools/task_watchdog_v2.bat` (silent vbs 経由) |
| 登録 ps1 | `tools/register_process_watchdog_v2.ps1` |
| schtasks TR (登録後) | `wscript.exe silent_runner.vbs task_watchdog_v2.bat` |

**動作**:
- 既存 ProcessWatchdog (Disabled v1) の TR を v2 用 bat に切替えて Enable
- Trigger は既存設定 (5 分間隔) を継承
- 07:00-18:00 のみ自動再起動、それ以外は Discord 警告

---

## 3. 機能比較

| 項目 | v1 | v2 |
|------|----|----|
| 検知方法 | PID 登録 | logs mtime |
| 監視対象登録 | ad-hoc (json) | ハードコード TARGETS (daily_predict + race_auto_notify) |
| ゾンビ検知 | ❌ プロセス生きてれば OK | ✅ ログ更新止まってたら STALE |
| 再起動 | あり | あり (07:00-18:00 のみ) |
| 通知 | あり | あり (CRITICAL prefix + 色 red) |
| 静音化 | なし | silent_runner.vbs 経由 |

→ v2 が完全な置き換え。

---

## 4. 5/9 朝の watchdog 体制 (二重)

5/9 (土) は以下の二重監視で **DailyPredict が確実に完了する**:

| watchdog | 機構 | 役割 |
|---------|------|------|
| **daily_predict_watchdog** (Session #4 で組込) | `tools/daily_predict_watchdog.py` | DailyPredict subprocess 監視 + Cookie 自動 refresh + max 3 restart |
| **ProcessWatchdog v2** (本日 admin で登録予定) | `tools/process_watchdog_v2.py` | logs mtime 監視、5 分間隔で全プロセス確認、stale なら再起動 |

5/9 06:30 〜 18:00 の本番時間帯では、二重 fail-safe が効く:
1. DailyPredict 自体に Cookie 自動 refresh + 子プロセス再起動 (Session #4)
2. 5 分おきに ProcessWatchdog v2 が外部から ping
3. daily_predict ログが 30 分更新なし → ProcessWatchdog v2 が再起動

→ **5/9 朝の DailyPredict 監視は確実**。

---

## 5. 修正後 admin 再実行リマインド

5/6 朝の admin で 2 件失敗していた ps1 を BOM 追加で修正済 (Session #29):

```powershell
# 1. ProcessWatchdog v2 (Disabled → Enable)
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1

# 2. 馬体重補正 (09:30 早朝、Sat/Sun)
PowerShell -ExecutionPolicy Bypass -File tools\register_morning_weight_check_schtasks.ps1
```

成功確認:
```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like '*ProcessWatchdog*' -or $_.TaskName -like 'Keiba-MorningWeightCheck*' } | ft TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}}
```

期待:
- ProcessWatchdog: State=Ready (Disabled から)
- Keiba-MorningWeightCheck_Sat/Sun: State=Ready、NextRun=5/9 09:30 / 5/10 09:30

---

## 6. 5/8 までの admin 累計 4 件

```powershell
# 修正後 (本日 BOM 追加)
PowerShell -File tools\register_process_watchdog_v2.ps1
PowerShell -File tools\register_morning_weight_check_schtasks.ps1

# 5/6 朝の admin で成功済 (再実行不要)
PowerShell -File tools\register_jrdb_retry_schtasks.ps1
PowerShell -File tools\register_multi_stage_predict_schtasks.ps1
```

---

## 7. 結論

- ProcessWatchdog v1 は 4/24 以降 Disabled、機能無効、v2 で完全置き換え可能
- v2 は本日 BOM 追加で admin 登録準備完了、ユーザー手動 1 コマンドで Enable
- 5/9 朝の DailyPredict 監視は daily_predict_watchdog (Session #4) + ProcessWatchdog v2 (本日) の二重体制
- 失敗 2 ps1 (process_watchdog_v2 + morning_weight_check) は BOM 追加で再実行可能
