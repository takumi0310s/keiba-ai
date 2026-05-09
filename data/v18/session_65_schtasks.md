# Session #65 C: schtasks 登録

## 1. 登録結果

```
TaskName:      \Keiba-PreRacePredict_Watchdog_5_9
Schedule:      MINUTE / 30 min interval
Start:         2026/05/09 13:00
Duration:      07:00:00 (16:00 まで自然停止可能、 watchdog logic で 全 R cover)
Action:        wscript.exe silent_runner.vbs pre_race_predict_runner.bat --check-next-1h
Next Run:      2026/05/09 13:30
Status:        Ready
```

PowerShell snippet (再現用):

```powershell
$base = "C:\Users\takum\keiba-ai"
$vbs  = "$base\tools\silent_runner.vbs"
$bat  = "$base\pre_race_predict_runner.bat"
$tr   = "wscript.exe `"$vbs`" `"$bat`" --check-next-1h"
schtasks /Create /TN "Keiba-PreRacePredict_Watchdog_5_9" /TR $tr `
         /SC MINUTE /MO 30 /SD 2026/05/09 /ST 13:00 /DU 0700:00 /F
```

`/SC MINUTE /MO 30` は Win11 24H2 で許可、 Admin 不要。

## 2. runner 実装

`pre_race_predict_runner.bat` (repo root):

```bat
@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if "%~1"=="" (
  python tools/stage2_predict.py --check-next-1h >> logs\stage2_predict.log 2>&1
) else (
  python tools/stage2_predict.py %* >> logs\stage2_predict.log 2>&1
)
```

silent_runner.vbs (既存) で console window 隠す。

## 3. dedup 動作確認

```bash
$ python tools/stage2_predict.py --race-id 202604010312 --no-discord
[skip dedup] 202604010312 already predicted at 2026-05-09T13:05:47.238085
```

cache JSON (`data/v18/pre_race_predict_cache_5_9.json`) で 1 R 1 通保証。 watchdog が 30 分毎に複数回 fire しても重複なし。

## 4. live test fire (deferred)

ユーザー指示の `schtasks /Run /TN` による即時 manual fire は **deferred**。 理由:
- 13:30+ 自然 fire 待ちで Discord 着信確認可
- 即 fire すると 5 R candidate 全部に Discord 送信 (現 13:08 時点で window 内 5 件)
- predict_one_race が 1 R 数分かかる、 schtasks の同時実行制御不安定

13:30 自然 fire で Discord 1 通以上届けば動作 OK 判定。 届かない場合は `logs/stage2_predict.log` を tail。

## 5. 干渉禁止確認

| 項目 | 状態 |
|------|------|
| 既存 schtasks 49 件 | 不変 (Session #61 9 件含む) |
| 新規 schtask | 1 件追加 (合計 50) |
| daily_predict.py 呼び出し | なし (Session #64 spam 再発防止 確認) |
| race_auto_notify.py 呼び出し | なし |
| ProcessWatchdog kill-switch | `data/v18/process_watchdog_v2.kill` 維持 |
| stage2_predict 自身の kill-switch | `data/v18/pre_race_predict.kill` を touch で停止 |
| V15 model file | 触らない |
| 5/9 投票方針 | 不変 (新潟 12R ¥700) |

## 6. 16:30 以降

`/DU 0700:00` で 20:00 まで動作するが、 16:30 時点で全 R 発走済 → cache 全 R 記録済 → dedup で skip 連続 → 副作用なし。
17:00 cumulative + 20:30 summary (Session #61) と独立稼働。

## 7. 緊急停止

```bash
touch data/v18/pre_race_predict.kill
```

stage2_predict.py main() / cmd_check_next_1h() 冒頭で check、 即 no-op exit。 schtasks は fire しても 1 秒で抜ける。
