# multi_stage_predict 運用 手順書

**作成**: 2026-05-06 朝活 (Session #28)

---

## 1. admin で 1 コマンド (5/8 までに必須)

```powershell
# 管理者として PowerShell を起動して実行
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_multi_stage_predict_schtasks.ps1
```

→ 6 タスク (土日 × 3 stage) が登録 + Ready 化。

## 2. 確認

```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-MultiStagePredict*' } | ft TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath).NextRunTime}}
```

期待: 6 task 全て State=Ready、NextRun が 5/9 当日。

## 3. 手動実行 (test 用)

```bash
# 5/9 朝の動作確認
python tools/multi_stage_predict.py --stage test10 --date 20260509 --dry-run
python tools/multi_stage_predict.py --stage race11_1450 --date 20260509 --dry-run
python tools/multi_stage_predict.py --stage race12_1545 --date 20260509 --dry-run
```

## 4. ロールバック

```powershell
PowerShell -ExecutionPolicy Bypass -File tools\register_multi_stage_predict_schtasks.ps1 -Rollback
```

→ 6 task 削除。

## 5. ログ確認

```bash
# 5/9 当日
ls logs/multi_stage_predict_*_20260509.log

# 内容
cat logs/multi_stage_predict_test10_20260509.log
cat logs/multi_stage_predict_race11_1450_20260509.log
cat logs/multi_stage_predict_race12_1545_20260509.log
```

## 6. 5/9 admin タスク 一覧 (累計 4 件)

```powershell
# 1. ProcessWatchdog v2
PowerShell -ExecutionPolicy Bypass -File tools\register_process_watchdog_v2.ps1

# 2. 馬体重補正 (09:30 早朝)
PowerShell -ExecutionPolicy Bypass -File tools\register_morning_weight_check_schtasks.ps1

# 3. JRDB AM 9:00 retry
PowerShell -ExecutionPolicy Bypass -File tools\register_jrdb_retry_schtasks.ps1

# 4. multi_stage_predict (本書、10:00/14:50/15:45)
PowerShell -ExecutionPolicy Bypass -File tools\register_multi_stage_predict_schtasks.ps1
```

→ 4 つすべて 5/8 までに admin 実行で 5/9 自動運用準備完了。
