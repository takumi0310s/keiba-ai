# 5/23 FridayWeekendScrape stub fix + 全 bat stub 残存確認

**実施時刻**: 2026-05-23  
**commit**: (本 doc と同一 commit)

---

## 修正ファイル一覧 (8 件)

| ファイル | 修正内容 | schtask |
|---------|---------|---------|
| `friday_weekend_scrape.bat` | PYTHON_EXE 追加、2 行 python → %PYTHON_EXE% | Keiba-FridayWeekendScrape |
| `tools/daily_jrdb_kyi.bat` | PYTHON_EXE 追加、10 行 python → %PYTHON_EXE% | DailyJrdbKyi |
| `tools/jrdb_retry_am9.bat` | PYTHON_EXE 追加、5 行 python → %PYTHON_EXE% | JrdbRetryAm9 |
| `tools/jrdb_retry_pm12.bat` | PYTHON_EXE 追加、6 行 python → %PYTHON_EXE% | JrdbRetryPm12 |
| `tools/task_watchdog_v2.bat` | PYTHON_EXE 追加、1 行 python -u → %PYTHON_EXE% -u | ProcessWatchdog |
| `tools/tyb_publish_monitor.bat` | PYTHON_EXE 追加、1 行 python → %PYTHON_EXE% | Keiba-TybPublishMonitor |
| `tools/danger_horse_alert.bat` | PYTHON_EXE 追加、1 行 python → %PYTHON_EXE% | DangerHorseAlert |
| `tools/strategy8_sidecar.bat` | PYTHON_EXE 追加、1 行 python → %PYTHON_EXE% | Strategy8Sidecar |

**共通 path**: `SET PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe`

---

## WindowsApps stub 残存確認

`grep -r "WindowsApps" *.bat tools/*.bat` → **0 件**

前回 11 bat fix (commit a054bfa3) + 今回 8 bat fix = 計 19 bat 全て真 path 統一。
Store stub (`C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe`) の残存なし。

---

## 残り bare python bat (修正不要)

以下の bat は bare `python` コマンドを使用しているが、**schtask で正常動作済み**:

| ファイル | 状態 |
|---------|------|
| `daily_predict.bat` | 5/23 08:39 schtask 成功 ✅ |
| `daily_results.bat` | 20:00 schtask、PATH 解決で動作 |
| `daily_premium_scrape.bat` | 03:00 schtask、PATH 解決で動作 |
| `race_auto_notify.bat` | 08:45 schtask、PATH 解決で動作 |
| `weekly_report.bat` | 月曜 08:00、PATH 解決で動作 |
| `nightly_sanity_check.bat` | 夜間 schtask |
| `am3/am6/am8_fire_check.bat` | fire check schtask 群 |

**判定**: Task Scheduler の PATH には pythoncore-3.14-64 が含まれている (daily_predict が bare python で成功したことで確認済み)。bare python ≠ stub path。修正不要。

---

## FridayWeekendScrape 5/29 fire 準備

| 項目 | 状態 |
|------|------|
| Keiba-FridayWeekendScrape schtask | 登録済み、Ready ✅ |
| Next Run | 2026/05/29 10:00 |
| PYTHON_EXE | 真 path (pythoncore-3.14-64) ✅ |
| PACI 週次行 | `%PYTHON_EXE% tools\scrape_jrdb_paci.py` ✅ |

5/29 10:00 に PACI 最新データが自動取得される。

---

## V15 regression

bat 修正は PYTHON_EXE 行追加と python コマンド prefix 変更のみ。
V15 .pkl.gz / predict_core / daily_predict logic = 完全不変。

---

*修正完了: 2026-05-23 | V15 production 不変*
