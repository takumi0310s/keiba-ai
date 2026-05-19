# F-4: 5/19 今夜 fire 監視 (2026-05-19)

## 背景
F-3 admin 完了 (9/9 schtask 全登録)。今夜 4 件が初 fire。
Discord 完了通知送信済 (14:15 相当、 tools/notify_done.py OK)。

## 今夜 fire スケジュール

| 時刻 | タスク名 | bat | 実 Python module | log 期待パス |
|------|---------|-----|-----------------|-------------|
| 17:00 | Keiba-AnomalyCheck-1700 | tools/keiba_anomaly_check_1700.bat | tools/anomaly_auto_detector.py | logs/keiba_anomaly_check_1700_20260519.log |
| 20:30 | Keiba-RaceNotifyLogV2Aggregator | tools/keiba_race_notify_log_v2_aggregator.bat | tools/race_notify_log_v2_aggregator.py | logs/keiba_race_notify_log_v2_aggregator_20260519.log |
| 21:00 | Keiba-CumulativeAudit-Daily | tools/keiba_cumulative_audit.bat | tools/daily_cumulative_audit.py | logs/keiba_cumulative_audit_20260519.log |
| 22:00 | Keiba-FeaturesIntegrity-Daily | tools/keiba_features_integrity.bat | tools/features_integrity_monitor.py | logs/keiba_features_integrity_20260519.log |

## bat 仕様メモ

### 17:00 — Keiba-AnomalyCheck-1700
- module: `tools/anomaly_auto_detector.py` (--time arg なし)
- log: `logs/keiba_anomaly_check_1700_YYYYMMDD.log`

### 20:30 — Keiba-RaceNotifyLogV2Aggregator
- module: `tools/race_notify_log_v2_aggregator.py --date YYYYMMDD`
- `%date%` から `yyyy/mm/dd` → `yyyymmdd` 変換して渡す
- DailyResultsEvening (20:00) 完了後 30 分で 3-phase log 集計
- log: `logs/keiba_race_notify_log_v2_aggregator_YYYYMMDD.log`

### 21:00 — Keiba-CumulativeAudit-Daily
- module: `tools/daily_cumulative_audit.py` (read-only drift detect)
- DailyResultsEvening の 1 時間後に起動
- log: `logs/keiba_cumulative_audit_YYYYMMDD.log`

### 22:00 — Keiba-FeaturesIntegrity-Daily
- module: `tools/features_integrity_monitor.py`
- log: `logs/keiba_features_integrity_YYYYMMDD.log`

## fire 後確認手順

各 fire 後に以下を実施:

```powershell
# 1. log ファイル生成確認
ls C:\Users\takum\keiba-ai\logs\keiba_*.log

# 2. 当日 log 確認 (例: 17:00 task)
Get-Content C:\Users\takum\keiba-ai\logs\keiba_anomaly_check_1700_20260519.log -Tail 20

# 3. エラーがないか確認
Select-String "ERROR|Traceback|Exception" C:\Users\takum\keiba-ai\logs\keiba_*20260519*.log
```

### Discord 確認
- #updates チャンネルに各 module からの通知が届いているか確認
- 通知なし = module 内の Discord 送信が失敗 or log のみ出力の module

### 異常時
```bash
# tools/admin_verify_v2.py があれば再実行
python tools/admin_verify_v2.py

# schtask 最終実行結果確認 (PowerShell 管理者)
# Get-ScheduledTask -TaskName "Keiba-*" | Get-ScheduledTaskInfo | Select TaskName, LastRunTime, LastTaskResult
```

## 5/23 SAT 08:30 LiveOrchestrator 初回 fire まで残りスケジュール

| 日時 | イベント |
|------|---------|
| 2026-05-19 17:00 | Keiba-AnomalyCheck-1700 初 fire |
| 2026-05-19 20:30 | Keiba-RaceNotifyLogV2Aggregator 初 fire |
| 2026-05-19 21:00 | Keiba-CumulativeAudit-Daily 初 fire |
| 2026-05-19 22:00 | Keiba-FeaturesIntegrity-Daily 初 fire |
| 2026-05-20〜22 | 平日 fire 継続監視 (異常なければ自動) |
| **2026-05-23 08:30** | **LiveOrchestrator 初 fire (本番 race day)** |

## 注意事項

- V15 .pkl.gz / predict_core / daily_predict / app.py は変更禁止
- schtask 操作は admin のみ (本ファイルは read-only reference)
- 5/23 LiveOrchestrator fire 前日 (5/22 夜) に deploy-check スキル実行推奨
- race_notify_log_v2_aggregator の `--date` は cmd `%date%` から自動取得 (ja-JP locale 対応済)

## 関連ファイル

| ファイル | 用途 |
|---------|------|
| tools/keiba_anomaly_check_1700.bat | 17:00 bat |
| tools/keiba_race_notify_log_v2_aggregator.bat | 20:30 bat |
| tools/keiba_cumulative_audit.bat | 21:00 bat |
| tools/keiba_features_integrity.bat | 22:00 bat |
| tools/anomaly_auto_detector.py | 17:00 実 module |
| tools/race_notify_log_v2_aggregator.py | 20:30 実 module |
| tools/daily_cumulative_audit.py | 21:00 実 module |
| tools/features_integrity_monitor.py | 22:00 実 module |
| docs/F-3_SCHTASK_REGISTER_LOG.md | F-3 admin 登録ログ (存在すれば) |
