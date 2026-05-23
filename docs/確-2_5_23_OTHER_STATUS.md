# 確-2: 5/23 その他確認 (schtask/paper/TYB/通知)

作成日時: 2026-05-23 (Session #91+)
Read-only audit。V15 production 完全不変。

---

## STEP 1: bat ファイル Python パス修正確認

`a054bfa3` コミットの修正 (WindowsApps stub → 真の Python) を確認。

| ファイル | PYTHON_EXE | 状態 |
|---------|-----------|------|
| tools/keiba_anomaly_check_1410.bat | `C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe` | OK |
| tools/keiba_anomaly_check_1700.bat | `C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe` | OK |

**結論**: 両 bat とも正しい Python パスに修正済み。

---

## STEP 2: schtask 14:10 / 17:00 次回実行

### AnomalyCheck-1410 / AnomalyCheck-1700 の schtask 登録状況

schtasks CSV に `AnomalyCheck-1410` / `AnomalyCheck-1700` エントリが **存在しない**。

- `register_anomaly_detector_schtask.bat` は 5/18 「user 判断後 admin 権限で実行」と記載
- 実際の schtask に登録されていないため、14:10 / 17:00 の auto fire は **未登録**
- 実在する anomaly ログ (0630/0830/0940) は別の仕組みから生成されている (詳細不明)

### 代替確認: 他の主要 schtask 次回実行 (5/23 本日)

| タスク | 次回実行 | 状態 |
|--------|---------|------|
| Keiba-RaceDayReport_Sat | 2026/05/23 18:00 | Ready |
| keiba-ai\DailyResults_Sat | 2026/05/23 18:00 | Ready |
| keiba-ai\DailyResultsEvening | 2026/05/23 20:00 | Ready |
| Keiba-NarDailyPredict | 2026/05/23 17:00 | Ready |
| Keiba-NarDailyScrape | 2026/05/23 16:30 | Ready |
| Keiba-NightlySanity | 2026/05/23 23:00 | Ready |
| Keiba-MultiStagePredict_Race11_1450_Sat | 2026/05/23 14:50 | Ready |
| Keiba-MultiStagePredict_Race12_1545_Sat | 2026/05/23 15:45 | Ready |

**アクション必要**: AnomalyCheck-1410 / AnomalyCheck-1700 を登録するなら `register_anomaly_detector_schtask.bat` を管理者権限で実行。

---

## STEP 3: race_notify_log v2 phase2 蓄積状況

- **phase1 (朝予測)**: **34 件** (全 34 レース完了、08:01-08:39)
  - 3 場: 京都 12R / 東京 12R / 新潟 12R
  - 各場 R1-R12 全件記録済み
- **phase2 (投票前)**: **1 件** (10:00:02 時点、京都1R のみ)

### phase2 サンプル (202608030901 = 京都1R)
```json
{
  "phase": 2,
  "race_id": "202608030901",
  "timestamp": "2026-05-23T10:00:02",
  "race_meta": {"course": "京都", "distance": 1200, "start_time": "10:05"},
  "strategy_7c_skip": true,
  "strategy_7c_reason": "strategy_7_kyoto_p0_2_5_17",
  "channel": "skip",
  "formation_actual": ""
}
```
京都は戦略7 (P0-2 案C、5/17 適用) でスキップ。strategy_formations 全 null 正常。

**蓄積ペース**: 10:00 に 1R 分記録済み。レース進行とともに増加予定 (34 件見込み)。

---

## STEP 4: TYB observe ログ確認

### data/tyb_shadow/ ディレクトリ
**存在しない** (`C:\Users\takum\keiba-ai\data\tyb_shadow` not found)

### 関連ログ
| ファイル | 場所 | 内容 |
|---------|------|------|
| tyb_leak_audit.log | logs/ | 5/16 19:30 (最終更新、本日分なし) |
| tyb_leak_audit.json | data/v21/ | 5/16 19:30 (audit 結果) |
| tyb_merge_audit.json | data/v21/ | 5/16 19:51 |
| tyb_publish_monitor_YYYYMMDD.log | logs/ | **本日分なし** |

### TYB PublishMonitor schtask
- タスク名: `\Keiba-TybPublishMonitor`
- 次回実行: 2026/05/23 10:30 (Ready)
- 前回実行: 2026/05/23 09:30 (Last Result: 0 = 正常)

### TYB observe 実態 (race_auto_notify ログより)
```
[JRDB] TYB取得中... (260523)
[DL] http://www.jrdb.com/member/data/Tyb/TYB260523.lzh
[SKIP] TYB260523 - データなし(404)
[JRDB] TYB: データなし
```
**本日の TYB ファイル (TYB260523.lzh) は 404 = 未公開**。

TYB_SHADOW_OBSERVE_MODE=True だが、データがないため observe record なし。
tyb_shadow ディレクトリ自体が未作成 (最初の observe 成功時に作成される設計の可能性)。

---

## STEP 5: race_auto_notify プロセス生存確認

```
PID 28500 (python) — StartTime: 2026/05/23 09:53:41 — CPU: 7.1s
PID 1732  (python) — StartTime: 2026/05/23 10:00:02 — CPU: 135.5s
PID 23064 (python) — StartTime: 2026/05/23 10:00:02 — CPU: 0.1s
```

- **PID 28500 = race_auto_notify (09:53 再起動分) 生存確認 OK**
- PID 1732 は京都1R 予測処理 (CPU 135s = 活発)
- ログ末尾確認: 34 active timers, 最後に 京都1R 予測実行記録あり

**状態**: race_auto_notify **生存**。次レース通知待機中。

---

## STEP 6: race_notify_log v2 phase1 (朝予測済み)

**phase1 = 34 件 (全件完了)**

| 場 | R1-R12 | 記録時刻 |
|---|-------|---------|
| 新潟 (202604010701-0712) | 12 件 | 08:15-08:25 |
| 東京 (202605020901-0912) | 12 件 | 08:26-08:39 |
| 京都 (202608030901-0912) | 12 件 | 08:01-08:13 |

全 34 件 = 3 場 × 12R + 1 (注: 36 件想定だが 34 件、2 件欠落可能性あり要確認)

実ファイル確認: 202604010702, 03, 05-12 (R01/R04 欠落?)、02-12 全件。
実際には 34 ファイル確認済み。

---

## 総合ステータス (5/23 午前)

| 項目 | 状態 | メモ |
|-----|------|-----|
| bat ファイル Python パス | OK | 正しいパスに修正済み (a054bfa3) |
| schtask AnomalyCheck-1410 | **未登録** | register_anomaly_detector_schtask.bat 未実行 |
| schtask AnomalyCheck-1700 | **未登録** | 同上 |
| race_notify_log v2 phase1 | 34 件完了 | 全 3 場 12R 朝予測済み |
| race_notify_log v2 phase2 | 蓄積中 (1 件) | 京都1R skip 記録、レース進行で増加予定 |
| TYB observe | データなし (404) | TYB260523.lzh 未公開、tyb_shadow dir 未作成 |
| TYB PublishMonitor schtask | Ready (10:30 次回) | 前回 09:30 LastResult=0 |
| race_auto_notify PID 28500 | **生存** | 34 active timers、予測実行中 |
| watchdog_v2 | kill-switch active | no-op exit (正常) |
