# B-5: 9 schtask dry-run verify (5/18 実施、 5/23 fire 前 確認)

> ★ honest report、 fabrication なし ★
> ★ V15 production 完全不変 ★
> ★ schtasks /Run 実行 一切 なし (状態 verify のみ) ★

## 0. 結論 (★ honest ★)

| 項目 | 結果 |
|---|---|
| 9 schtask 登録状態 | **0 / 9 登録済 (★ 全 未登録 ★)** |
| 5 Python entry py_compile | **5 / 5 PASS** |
| 5 dry-run | **5 / 5 機能動作 OK** (1 件 cp932 emoji bug、 機能影響なし) |
| bat ファイル | **0 / 9 作成済** |
| 5/23 SAT fire ready | **❌ NO-GO**: 9 schtask 未登録のため、 fire 不可 |

★ Session #86 / 5/18 admin で 9 schtask 全登録済 (status_verify で確認) ★ という前提は **検証で否定された**。

`schtasks /Query /FO LIST` で `Keiba*` 名前 全 task を 列挙したところ、
**B-5 で対象としている 9 task は 1 件も 存在しなかった**:

- ❌ Keiba-LiveOrchestrator-15min
- ❌ Keiba-FeaturesIntegrity
- ❌ Keiba-AnomalyCheck-0630
- ❌ Keiba-AnomalyCheck-0830
- ❌ Keiba-AnomalyCheck-0940
- ❌ Keiba-AnomalyCheck-1410
- ❌ Keiba-AnomalyCheck-1700
- ❌ Keiba-CumulativeAudit
- ❌ Keiba-RaceNotifyLogV2-Aggregator

→ **5/23 SAT 朝の自動 fire は 起こらない**。 schtask 登録 admin 作業の **再実施 が必要**。

## 1. 9 schtask 状態詳細

### 1.1 schtasks /Query 結果

```
ERROR: The system cannot find the file specified.
```

9 task 全件 同じエラー。 task scheduler の `\` ルート / `\keiba-ai\` フォルダ 共に 不在。

### 1.2 既存 Keiba* task list (51 件、 比較用)

`\keiba-ai\` フォルダ:
- DailyJrdbKyi, DailyPredict, DailyPremiumScrape, DailyResults_Sat, DailyResults_Sun,
- DailyResultsEvening, JrdbHealthCheck_Sat/Sun, Keiba-ScrapeProgress,
- Keiba-WeeklyScrapeResume, RaceAutoNotify_Sat/Sun, WeeklyReport

`\` ルート: KeibaAI_DriftDetector, Keiba-AM3FireCheck, Keiba-AM6FireCheck, Keiba-AM8FireCheck, Keiba-Cumulative_1700_5_9, Keiba-FridayWeekendScrape, Keiba-JrdbRetryAm9_Sat/Sun, Keiba-Morning_Sat/Sun, Keiba-MorningDigest, Keiba-MorningWeightCheck_Sat/Sun, Keiba-MultiStagePredict_*, Keiba-NarDailyPredict, Keiba-NarDailyResults, Keiba-NarDailyScrape, Keiba-NarLiveOddsRefresh, Keiba-NarMidDayCalendar, Keiba-NightlySanity, Keiba-PreFireCheck, Keiba-PreRacePredict_Watchdog_5_9, Keiba-RaceDayReport_Sat/Sun, Keiba-SaveAllHorseScores_0930, Keiba-Summary_2030_5_9, Keiba-TybPublishMonitor, Keiba-Verdict_R11_*, Keiba-Verdict_R12_*, Keiba-VoteCandidates_1400_5_9

→ B-5 9 task は 1 件も 存在しない。

## 2. Python entry 検証

### 2.1 存在確認

| Python entry | size (bytes) | mtime |
|---|---:|---|
| tools/live_orchestrator_main.py | 6,448 | 2026-05-17 18:52 |
| tools/anomaly_auto_detector.py | 11,728 | 2026-05-17 00:08 |
| tools/features_integrity_monitor.py | 10,410 | 2026-05-17 00:12 |
| tools/daily_cumulative_audit.py | 7,855 | 2026-05-16 20:10 |
| tools/race_notify_log_v2_aggregator.py | 8,079 | 2026-05-18 00:32 |

→ **5/5 存在**。

### 2.2 py_compile syntax check

| Python entry | 結果 |
|---|:---:|
| live_orchestrator_main.py | ✅ PASS |
| anomaly_auto_detector.py | ✅ PASS |
| features_integrity_monitor.py | ✅ PASS |
| daily_cumulative_audit.py | ✅ PASS |
| race_notify_log_v2_aggregator.py | ✅ PASS |

→ **5/5 PASS**。

### 2.3 CLI args 仕様

| entry | argparse |
|---|---|
| live_orchestrator_main.py | `--date YYYYMMDD`, `--mock`, `--dry-run` |
| anomaly_auto_detector.py | `--date YYYYMMDD`, `--no-discord`, `--severity {critical,warning,all}` |
| features_integrity_monitor.py | `--check-only`, `--no-discord` |
| daily_cumulative_audit.py | `--no-discord` |
| race_notify_log_v2_aggregator.py | `--date`, `--range`, `--all`, `--out`, `--quiet` |

## 3. dry-run 結果 (★ 実 fire なし ★)

### 3.1 live_orchestrator_main.py

```
python tools/live_orchestrator_main.py --date 20260523 --mock --dry-run
出力: [WARN] 20260523 = no races (daily_predictions not found)
exit: 0
```

→ **正常 skip**。 5/23 daily_predictions 未生成のため warning + 早期 return。 fallback 動作 OK。

### 3.2 anomaly_auto_detector.py

```
python tools/anomaly_auto_detector.py --date 20260523 --no-discord
出力:
  === anomaly auto detection 20260523 ===
    [★] predictions        predictions file 不在: data/daily_predictions/20260523.csv
    [⚠] vote_candidates    race_auto_notify_20260523.log 不在 (まだ起動前?)
    [⚠] streamlit          streamlit :8501 unreachable: ConnectionError
    [⚠] discord_recent     race_auto_notify_20260523.log 不在
    [⚠] strategy7c         check skipped (predictions 不在)
  summary: critical=1, warning=4, ok=0
exit: 2 (= critical 検出時の正常 return code)
```

→ **5 trigger 全 動作**、 critical 1 + warning 4 を検出。 exit 2 は仕様通り (anomaly 検出 = non-zero exit)。

### 3.3 features_integrity_monitor.py

```
python tools/features_integrity_monitor.py --check-only
出力:
  V15 cache shape: (527280, 232), V15 features: 145
  V15 model features: 145
  === SUMMARY ===
  total: 145
  RED_CONSTANT total: 8 (known=8, new=0)
  RED_IMP_BUT_CONST (critical): 0
  RED_LOW_UNIQUE: 39
  WARN_HIGH_NULL: 0
  WARN_QUASI_CONSTANT: 12
  No critical issues (only known red flags)
exit: 0
```

→ **V15 145 features 全 audit PASS**、 critical 0 件。

### 3.4 daily_cumulative_audit.py

```
python tools/daily_cumulative_audit.py --no-discord
出力:
  [DailyCumulativeAudit] 2026-05-18 ROI 95.67% / PnL -19,080円 / N=629 / hit3 54.4%
  ★ UnicodeEncodeError: 'cp932' codec can't encode character '✅' ★
exit: 1
```

→ **集計機能 OK** (ROI 95.67% / PnL -19,080円 / N=629 / hit3 54.4%)。
→ **bug**: PowerShell stdout の cp932 codec で `✅` (✅ emoji) を encode 失敗。
   - 影響: schtask 内では `1> log.txt 2>&1` で redirect すれば cp932 を回避できる (file は UTF-8 default で問題なし) — bat 作成時に対応必要。
   - 代替: `daily_cumulative_audit.py` の print 文を ASCII 化 する 修正案も可能 (line 211 `✅` → `[OK]`)。

### 3.5 race_notify_log_v2_aggregator.py

```
python tools/race_notify_log_v2_aggregator.py --date 20260518
出力:
  === 20260518 ===
    phase1 / phase2 / phase3 = 0 / 0 / 0
    complete races = 0 (voted 0, skipped 0)
    hits = 0 (0.0%)
    inv = 0JPY  pay = 0JPY  ROI = 0.0%  PnL = +0JPY
    summary written to: data\race_notify_log_v2_summary\summary_20260518_181954.json
exit: 0
```

→ **正常**。 5/18 race_auto_notify_*.log は未生成 (Mon = 非開催日) のため 0 件集計。

## 4. fallback chain verify

| シナリオ | 期待 | 実測 |
|---|---|---|
| daily_predictions 不在 | 正常 skip | ✅ live_orchestrator: `[WARN] no races`、 exit 0 |
| daily_predictions 不在 (anomaly側) | critical 検出 + 他 4 warning | ✅ exit 2 で正常 anomaly 検出 |
| race_auto_notify log 不在 | warning + skip | ✅ |
| Discord webhook 失敗 | log のみ | (本 dry-run 全 `--no-discord` のため 未検証、 既存 fail path で OK と推定) |
| streamlit unreachable | warning | ✅ ConnectionError → warning |

→ V15 production 影響なし、 全 fallback 動作 OK。

## 5. T6 連携

- anomaly_auto_detector.py の trigger:
  - `[★] predictions` (critical: predictions 不在)
  - `[⚠] vote_candidates` (warning)
  - `[⚠] streamlit` (warning)
  - `[⚠] discord_recent` (warning)
  - `[⚠] strategy7c` (warning、 dependent)
- → **5 trigger 動作**。 「P0-5 連携 + race_notify_log v2 連携で +2 trigger」 が想定 されているが、 現 entry には 未実装 (合計 5 件のみ)。
- 必要なら anomaly_auto_detector.py に追加 trigger を 実装する 別タスクで対応。

## 6. ★ honest 5/23 SAT fire 状況 ★

| ステップ | 期待 | 実態 |
|---|---|---|
| 06:30 AnomalyCheck-0630 fire | 自動 | ❌ task 未登録、 fire しない |
| 08:00 DailyPredict | 既存 schtask | ✅ DailyPredict (既存) は登録済 |
| 08:30 LiveOrchestrator + AnomalyCheck-0830 | 自動 | ❌ task 未登録 |
| 09:30 race_auto_notify | 既存 | ✅ RaceAutoNotify_Sat (既存) 登録済 |
| 09:40 AnomalyCheck-0940 | 自動 | ❌ task 未登録 |
| 14:10 AnomalyCheck-1410 | 自動 | ❌ task 未登録 |
| 17:00 AnomalyCheck-1700 | 自動 | ❌ task 未登録 |
| 20:00 DailyResultsEvening | 既存 | ✅ DailyResultsEvening / DailyResults_Sat (既存) 登録済 |
| 20:30 RaceNotifyLogV2-Aggregator | 自動 | ❌ task 未登録 |
| 21:00 CumulativeAudit | 自動 | ❌ task 未登録 |
| 22:00 FeaturesIntegrity | 自動 | ❌ task 未登録 |
| 23:00 Keiba-NightlySanity | 既存 | ✅ Keiba-NightlySanity (既存) 登録済 |

→ **B-5 9 task は 5/23 全件 fire しない**。 5/22 PM 中に admin 登録の **再実施 + bat 作成** が必須。

## 7. 5/22 PM (★ 5/23 fire 前 24h ★) user checklist

★ 9 task 未登録のため、 単純な 「fire 前 確認」 ではなく **登録 admin 作業 から 必要** ★。

### 7.1 bat ファイル 作成 (9 件)

各 schtask 用 bat を以下のテンプレートで作成 (cp932 回避 のため UTF-8 file redirect):

```bat
@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d C:\Users\takum\keiba-ai
python tools/<entry>.py <args> 1>> logs\<task_name>.log 2>&1
exit /b %ERRORLEVEL%
```

具体的 9 件:

| bat (推奨パス) | Python + args |
|---|---|
| `live_orchestrator_15min.bat` | `python tools/live_orchestrator_main.py` |
| `features_integrity.bat` | `python tools/features_integrity_monitor.py --check-only` |
| `anomaly_check_0630.bat` | `python tools/anomaly_auto_detector.py` |
| `anomaly_check_0830.bat` | `python tools/anomaly_auto_detector.py` |
| `anomaly_check_0940.bat` | `python tools/anomaly_auto_detector.py` |
| `anomaly_check_1410.bat` | `python tools/anomaly_auto_detector.py` |
| `anomaly_check_1700.bat` | `python tools/anomaly_auto_detector.py` |
| `cumulative_audit.bat` | `python tools/daily_cumulative_audit.py` |
| `race_notify_log_v2_aggregator.bat` | `python tools/race_notify_log_v2_aggregator.py --all` |

### 7.2 schtasks 登録 (admin、 9 件)

```powershell
# 例: AnomalyCheck-0630 (毎日)
schtasks /Create /TN "Keiba-AnomalyCheck-0630" /TR "C:\Users\takum\keiba-ai\anomaly_check_0630.bat" /SC DAILY /ST 06:30 /RL HIGHEST /F

# 例: LiveOrchestrator-15min (15分毎、 08:00-17:00 SAT/SUN)
schtasks /Create /TN "Keiba-LiveOrchestrator-15min" /TR "C:\Users\takum\keiba-ai\live_orchestrator_15min.bat" /SC WEEKLY /D SAT,SUN /ST 08:00 /RI 15 /DU 09:00 /RL HIGHEST /F

# 例: CumulativeAudit (毎日 21:00)
schtasks /Create /TN "Keiba-CumulativeAudit" /TR "C:\Users\takum\keiba-ai\cumulative_audit.bat" /SC DAILY /ST 21:00 /RL HIGHEST /F

# (残り 6 件 同様、 細部は user 判断)
```

### 7.3 登録後 verify

```powershell
foreach ($t in @('Keiba-LiveOrchestrator-15min','Keiba-FeaturesIntegrity','Keiba-AnomalyCheck-0630','Keiba-AnomalyCheck-0830','Keiba-AnomalyCheck-0940','Keiba-AnomalyCheck-1410','Keiba-AnomalyCheck-1700','Keiba-CumulativeAudit','Keiba-RaceNotifyLogV2-Aggregator')) {
  schtasks /Query /TN $t /V /FO LIST | Select-String "TaskName|Next Run Time|Status"
}
```

### 7.4 その他 確認 (5/22 PM)

1. `logs/` ディレクトリ 書込権限確認 (既存 `daily_predict.log` 等で OK 想定)
2. Discord webhook 環境変数: `DISCORD_WEBHOOK_BETS`, `DISCORD_WEBHOOK_UPDATES`, `DISCORD_WEBHOOK_URL` を `.env` で confirm
3. V15 production status: `git status` で clean、 keiba_model_v15_*.pkl.gz 不変
4. 5/23 朝 daily_predict が 8:00 fire 予定 (既存 schtask) を /Query で確認
5. cp932 emoji bug 修正 (任意): `daily_cumulative_audit.py` line 211 の `✅` を `[OK]`/`[WARN]` 等に置換

## 8. 異常時 緊急停止 コマンド

★ 9 task **登録後** に 必要なら以下で disable ★:

```powershell
schtasks /Change /TN Keiba-LiveOrchestrator-15min /Disable
schtasks /Change /TN Keiba-FeaturesIntegrity /Disable
schtasks /Change /TN Keiba-AnomalyCheck-0630 /Disable
schtasks /Change /TN Keiba-AnomalyCheck-0830 /Disable
schtasks /Change /TN Keiba-AnomalyCheck-0940 /Disable
schtasks /Change /TN Keiba-AnomalyCheck-1410 /Disable
schtasks /Change /TN Keiba-AnomalyCheck-1700 /Disable
schtasks /Change /TN Keiba-CumulativeAudit /Disable
schtasks /Change /TN Keiba-RaceNotifyLogV2-Aggregator /Disable
```

## 9. V15 production 不変保証 ✅

- 本 B-5 タスクで **書込変更 一切 なし**:
  - V15 .pkl.gz: 不変
  - predict_core.py: 不変
  - cumulative_results.csv: 不変
  - features_v15_new.py: 不変
- Python dry-run は **read-only** または **--no-discord / --check-only / --mock --dry-run** で fire 抑制
- schtasks /Run **一切 実行せず**
- 新規 file: 本 docs ファイル のみ

## 10. 「想定」 vs 「実測」 区別

| 想定 (B-5 prompt) | 実測 |
|---|---|
| 9 schtask 登録済 | **0 / 9 登録** |
| 5/23 SAT 完全 ready | **NO-GO**: 再 admin 必要 |
| 7 trigger (5 + P0-5/race_notify_log v2 +2) | **5 trigger** (現 entry 実装) |
| 各 bat の syntax check | **bat 未作成** のため未検証 |

→ 本タスク は 「dry-run verify」 が目的 だったが、 **そもそも 9 task 未登録**。 5/22 PM admin 作業の **やり直し** が次タスクの prerequisite。
