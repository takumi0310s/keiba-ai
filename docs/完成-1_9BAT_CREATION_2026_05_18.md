# 完成-1: 9 bat 作成 + dry-run PASS verify (2026-05-18)

## 背景
5/18 B-5 で 9 schtask 全未登録 確定。 5/22 PM admin で 5/24 fire ready のため、
本 sub-task で 9 bat 全作成 + dry-run verify。

## 絶対遵守 status
- V15 production 不変: OK (predict_core / daily_predict / race_auto_notify / app.py / .pkl.gz 一切変更なし)
- destructive op: なし
- schtasks /Create 実行: なし (admin manual のみ、 本 sub-task 範囲外)
- caveman mode: 適用

---

## 9 bat status table

| # | bat path | status | 対応 python module | python 存在 | py_compile |
|---|----------|--------|--------------------|-------------|-----------|
| 1 | tools/live_orchestrator.bat | 既存 (5/17 4-A) | tools/live_orchestrator_main.py | OK | PASS |
| 2 | tools/keiba_features_integrity.bat | 新規作成 | tools/features_integrity_monitor.py | OK | PASS |
| 3 | tools/keiba_anomaly_check_0630.bat | 新規作成 | tools/anomaly_auto_detector.py | OK | PASS |
| 4 | tools/keiba_anomaly_check_0830.bat | 新規作成 | tools/anomaly_auto_detector.py | OK | PASS |
| 5 | tools/keiba_anomaly_check_0940.bat | 新規作成 | tools/anomaly_auto_detector.py | OK | PASS |
| 6 | tools/keiba_anomaly_check_1410.bat | 新規作成 | tools/anomaly_auto_detector.py | OK | PASS |
| 7 | tools/keiba_anomaly_check_1700.bat | 新規作成 | tools/anomaly_auto_detector.py | OK | PASS |
| 8 | tools/keiba_cumulative_audit.bat | 新規作成 | tools/daily_cumulative_audit.py | OK | PASS |
| 9 | tools/keiba_race_notify_log_v2_aggregator.bat | 新規作成 | tools/race_notify_log_v2_aggregator.py | OK | PASS |

★ honest 訂正 (仕様書 vs 実態) ★

| 仕様書 module 名 | 実 module 名 | 理由 |
|------------------|--------------|------|
| tools/T1_features_integrity_check.py | tools/features_integrity_monitor.py | T1_*.py は repo 内 未存在、 実装は features_integrity_monitor.py (Session 既存) |
| tools/T6_anomaly_check.py --time HHMM | tools/anomaly_auto_detector.py (--time arg なし) | T6_*.py は repo 内 未存在、 実装は anomaly_auto_detector.py。 仕様 --time HHMM は実装されていないため、 時刻別の差別化は schtask /ST のみで実現 (既存 register_anomaly_detector_schtask.bat と同方針)。 module 側の --time 実装は 別 sub-task に委譲 |

---

## dry-run check

| # | bat | cd /d count | python count | dry-run |
|---|-----|-------------|--------------|---------|
| 1 | live_orchestrator.bat | 1 | 1 | PASS |
| 2 | keiba_features_integrity.bat | 1 | 1 | PASS |
| 3 | keiba_anomaly_check_0630.bat | 1 | 1 | PASS |
| 4 | keiba_anomaly_check_0830.bat | 1 | 1 | PASS |
| 5 | keiba_anomaly_check_0940.bat | 1 | 1 | PASS |
| 6 | keiba_anomaly_check_1410.bat | 1 | 1 | PASS |
| 7 | keiba_anomaly_check_1700.bat | 1 | 1 | PASS |
| 8 | keiba_cumulative_audit.bat | 1 | 1 | PASS |
| 9 | keiba_race_notify_log_v2_aggregator.bat | 1 | 1 | PASS |

dry-run PASS = 9/9

logs/ ディレクトリ: 既存 (mkdir 不要、 但し各 bat 内に `if not exist logs mkdir logs` 防御あり)

---

## 5/22 admin /Create 用 path list (本 sub-task 外で実行)

```cmd
:: Windows admin cmd で 9 件 /Create 想定

:: 1. LiveOrchestrator (WEEKLY SAT,SUN 08:30)
schtasks /Create /TN "Keiba-LiveOrchestrator-15min" ^
  /TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat" ^
  /SC WEEKLY /D SAT,SUN /ST 08:30 /RL HIGHEST /F

:: 2. FeaturesIntegrity (DAILY 22:00)
schtasks /Create /TN "Keiba-FeaturesIntegrityCheck" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_features_integrity.bat" ^
  /SC DAILY /ST 22:00 /F

:: 3-7. AnomalyCheck × 5 (DAILY 06:30 / 08:30 / 09:40 / 14:10 / 17:00)
schtasks /Create /TN "Keiba-AnomalyCheck-0630" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0630.bat" ^
  /SC DAILY /ST 06:30 /F
schtasks /Create /TN "Keiba-AnomalyCheck-0830" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0830.bat" ^
  /SC DAILY /ST 08:30 /F
schtasks /Create /TN "Keiba-AnomalyCheck-0940" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0940.bat" ^
  /SC DAILY /ST 09:40 /F
schtasks /Create /TN "Keiba-AnomalyCheck-1410" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1410.bat" ^
  /SC DAILY /ST 14:10 /F
schtasks /Create /TN "Keiba-AnomalyCheck-1700" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1700.bat" ^
  /SC DAILY /ST 17:00 /F

:: 8. CumulativeAudit (DAILY 21:00)
schtasks /Create /TN "Keiba-CumulativeAudit" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_cumulative_audit.bat" ^
  /SC DAILY /ST 21:00 /F

:: 9. RaceNotifyLogV2 Aggregator (DAILY 20:30)
schtasks /Create /TN "Keiba-RaceNotifyLogV2-Aggregator" ^
  /TR "C:\Users\takum\keiba-ai\tools\keiba_race_notify_log_v2_aggregator.bat" ^
  /SC DAILY /ST 20:30 /RL HIGHEST /F
```

注: 既存の register_*_schtask.bat (anomaly_auto_detector / features_integrity / race_notify_log_v2_aggregator / p0_5) は
旧 bat path を参照する。 本完成-1 で作成した keiba_*.bat 名と二重登録回避のため、
admin 実行時は **新 keiba_*.bat path** を使用するか、 旧 register script を使用するか
の片方を選ぶ (両方走らせると schtask 名衝突)。 推奨は新 keiba_*.bat path で統一。

---

## 統一 template (全 8 新規 bat)

```bat
@echo off
:: ============================================================
:: <task name> (<schedule>)
:: ★ 完成-1 sub-task (5/18) で作成 ★
:: ============================================================
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
python -u tools\<module>.py [args] >> logs\<task>_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %ERRORLEVEL%
```

live_orchestrator.bat (既存) は logs/ → data/live_orchestrator_log/stdout.log の旧形式を維持
(変更すると 5/17 4-A の verify 結果が無効化される懸念のため不変)。

---

## 後続 sub-task 委譲事項 (honest 報告)

1. tools/T1_features_integrity_check.py / T6_anomaly_check.py の rename 検討
   (但し既存実装で機能は満たすため、 純粋 cosmetic、 必須ではない)
2. anomaly_auto_detector.py に --time HHMM arg 追加
   (現状 5 schtask が同 module を呼ぶ → log で区別できる only)
3. 5/22 admin schtask /Create 実行 (本 sub-task 範囲外)
4. live_orchestrator.bat の log path を logs/ 統一する整理 (現状 data/live_orchestrator_log/)

---

完成-1 完了、 9 bat 全作成、 dry-run 9/9 PASS
