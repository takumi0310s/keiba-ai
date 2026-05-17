# 5/18 朝 admin 作業手順 (★ 寝起き 1 doc ★)

> **目的**: 5/18 朝に admin 権限で schtask 4 件 (計 8 task) を 1 回で登録する。
> **対象**: takumi 本人、 Windows PowerShell (管理者として実行) で実施。
> **所要**: 5-10 分 (登録 1-2 分 + 動作確認 3-5 分)。
> **絶対不変**: V15 / predict_core / 既存 schtasks。 admin 操作は本 doc の手順 のみ。

---

## 0. 前提 (★ 5/17 G1 day 終了後の朝 ★)

- 5/17 (土) G1 day 終了済
- 5/18 (日) 朝に admin schtask 登録 → 同日 08:30 から LiveOrchestrator 初回 fire
- 5/18 (日) も開催あり → 即 paper eval 開始可能

---

## 1. 登録対象 schtask (4 件 / 計 8 task)

| # | schtask 名 | timing | bat | priority | admin |
|---|-----------|--------|-----|----------|-------|
| 1 | Keiba-LiveOrchestrator-15min | WEEKLY SAT/SUN 08:30 | `tools/p0_5_schtask_register.bat` | 高 | 必須 |
| 2 | Keiba-FeaturesIntegrityCheck | DAILY 22:00 | `tools/register_features_integrity_schtask.bat` | 中 | 必須 |
| 3 | Keiba-AnomalyCheck-* (5 件) | DAILY 06:30 / 08:30 / 09:40 / 14:10 / 17:00 | `tools/register_anomaly_detector_schtask.bat` | 高 | 必須 |
| 4 | Keiba-DailyCumulativeAudit | DAILY 21:00 | `tools/register_daily_cumulative_audit_schtask.bat` | 中 | 必須 |

**bat 実在確認 (5/17 完了)**: 4 ファイル全て tools/ 配下に存在確認済。

---

## 2. 各 schtask の詳細

### 2-1. Keiba-LiveOrchestrator-15min (★ 最重要 ★)

| 項目 | 値 |
|------|----|
| schtask 名 | `Keiba-LiveOrchestrator-15min` |
| timing | WEEKLY SAT,SUN 08:30 |
| 実行 cmd | `C:\Users\takum\keiba-ai\tools\live_orchestrator.bat` |
| run level | HIGHEST (admin) |
| 初回 fire | 5/18 (SUN) 08:30 — 登録当日に即 fire |
| 動作確認 | `schtasks /Query /TN "Keiba-LiveOrchestrator-15min" /V /FO LIST` |
| rollback | `schtasks /Delete /TN "Keiba-LiveOrchestrator-15min" /F` |

★ 既存 conflict: なし (新規 schtask)。
★ register bat 内で既登録時の上書き確認 prompt あり。

### 2-2. Keiba-FeaturesIntegrityCheck

| 項目 | 値 |
|------|----|
| schtask 名 | `Keiba-FeaturesIntegrityCheck` |
| timing | DAILY 22:00 |
| 実行 cmd | `python tools/features_integrity_monitor.py >> logs/features_integrity_YYYYMMDD.log 2>&1` |
| run level | (default、 admin 不要だが register に admin 必須) |
| 初回 fire | 5/18 (SUN) 22:00 |
| 動作確認 | `schtasks /Query /TN "Keiba-FeaturesIntegrityCheck"` |
| rollback | `schtasks /Delete /TN "Keiba-FeaturesIntegrityCheck" /F` |

★ 既存 conflict: なし。 22:00 は他 task 空き枠。
★ read-only モード、 red flag 検出時のみ Discord 通知。

### 2-3. Keiba-AnomalyCheck-* (5 件)

| schtask 名 | timing | 目的 |
|-----------|--------|------|
| Keiba-AnomalyCheck-0630 | DAILY 06:30 | DailyJrdbKyi 完了 (06:00) 直後の check |
| Keiba-AnomalyCheck-0830 | DAILY 08:30 | DailyPredict 完了 (08:00) 後の check |
| Keiba-AnomalyCheck-0940 | DAILY 09:40 | 09:30 Discord 通知 後の confirm (critical) |
| Keiba-AnomalyCheck-1410 | DAILY 14:10 | 14:00 投票準備直前 check |
| Keiba-AnomalyCheck-1700 | DAILY 17:00 | 開催後 evening 整理 check |

| 共通項目 | 値 |
|---------|----|
| 実行 cmd | `C:\Users\takum\keiba-ai\tools\anomaly_auto_detector.bat` |
| run level | (default) |
| 初回 fire | 5/19 (MON) 06:30 から (登録当日の 5/18 にも timing によっては fire) |
| 動作確認 | `schtasks /Query /TN "Keiba-AnomalyCheck-0630"` (他 4 件も同様) |
| rollback | `schtasks /Delete /TN "Keiba-AnomalyCheck-{0630,0830,0940,1410,1700}" /F` (5 件) |

★ 既存 conflict: なし。
★ /F flag で 同名 task 上書き登録 (既存があれば差し替え)。

### 2-4. Keiba-DailyCumulativeAudit

| 項目 | 値 |
|------|----|
| schtask 名 | `Keiba-DailyCumulativeAudit` |
| timing | DAILY 21:00 |
| 実行 cmd | `C:\Users\takum\keiba-ai\tools\daily_cumulative_audit.bat` |
| run level | HIGHEST (admin) |
| 初回 fire | 5/18 (SUN) 21:00 |
| 動作確認 | `schtasks /Query /TN "Keiba-DailyCumulativeAudit"` |
| rollback | `schtasks /Delete /TN "Keiba-DailyCumulativeAudit" /F` |

★ 既存 conflict: なし (DailyResultsEvening 20:00 完了の 1h 後)。
★ logs/daily_cumulative_audit.log に出力。

---

## 3. 登録手順 (step-by-step)

### Step 1: Windows PowerShell (★ 管理者として実行 ★)

スタートメニュー → 「Windows PowerShell」右クリック → 「管理者として実行」。

```powershell
cd C:\Users\takum\keiba-ai
```

### Step 2: 順次実行 (各 5-30 sec)

```powershell
# 2-1. LiveOrchestrator (★ 最重要 ★)
.\tools\p0_5_schtask_register.bat

# 2-2. FeaturesIntegrity
.\tools\register_features_integrity_schtask.bat

# 2-3. AnomalyCheck (5 件 一括)
.\tools\register_anomaly_detector_schtask.bat

# 2-4. CumulativeAudit
.\tools\register_daily_cumulative_audit_schtask.bat
```

★ 各 bat の戻り値を必ず確認。 ERROR が出たら 次に進まず Section 8 rollback へ。

### Step 3: 登録確認

```powershell
schtasks /Query /FO LIST | findstr "Keiba"
```

期待 output: 既存 (DailyJrdbKyi/DailyPredict/MorningWeightCheck/DailyResultsEvening/NightlySanity etc.) + 新規 8 件 (LiveOrchestrator/FeaturesIntegrity/AnomalyCheck×5/CumulativeAudit) = 計 13+件。

各 task の Next Run Time も確認:
```powershell
schtasks /Query /TN "Keiba-LiveOrchestrator-15min" /V /FO LIST | findstr "Next"
schtasks /Query /TN "Keiba-FeaturesIntegrityCheck" /V /FO LIST | findstr "Next"
schtasks /Query /TN "Keiba-AnomalyCheck-0630" /V /FO LIST | findstr "Next"
schtasks /Query /TN "Keiba-DailyCumulativeAudit" /V /FO LIST | findstr "Next"
```

### Step 4: Discord 手動通知

`#アップデート` channel に投稿:
```
5/18 admin schtask 登録完了
- Keiba-LiveOrchestrator-15min (SAT/SUN 08:30)
- Keiba-FeaturesIntegrityCheck (DAILY 22:00)
- Keiba-AnomalyCheck-* (DAILY 06:30/08:30/09:40/14:10/17:00)
- Keiba-DailyCumulativeAudit (DAILY 21:00)
初回 fire: 本日 08:30 LiveOrchestrator
```

---

## 4. 既存 schtasks との conflict check

| 時刻 | task | 種別 |
|------|------|------|
| 03:00 | DailyPremiumScrape | 既存 |
| 06:00 | DailyJrdbKyi | 既存 |
| **06:30** | **★ AnomalyCheck-0630 (新規) ★** | DailyJrdbKyi 完了直後 |
| 07:30 | JrdbHealthCheck_Sat/Sun | 既存 (SAT/SUN) |
| 08:00 | DailyPredict | 既存 |
| **08:30** | **★ LiveOrchestrator-15min (新規 SAT/SUN) + AnomalyCheck-0830 (新規 DAILY) ★** | DailyPredict 完了後 |
| 08:45 | RaceAutoNotify | 既存 (SAT/SUN) |
| 09:30 | MorningWeightCheck | 既存 |
| **09:40** | **★ AnomalyCheck-0940 (新規) ★** | Discord 通知 confirmation |
| **14:10** | **★ AnomalyCheck-1410 (新規) ★** | 14:00 投票確定通知後 |
| **17:00** | **★ AnomalyCheck-1700 (新規) ★** | 開催後集計 |
| 18:00 | DailyResults (SAT/SUN) | 既存 |
| 20:00 | DailyResults / DailyResultsEvening | 既存 |
| **21:00** | **★ CumulativeAudit (新規) ★** | DailyResults 完了後 |
| **22:00** | **★ FeaturesIntegrity (新規) ★** | 当日 cache update 後 |
| 23:00 | Keiba-NightlySanity | 既存 |

★ **conflict なし** verify 済。 各 timing で 既存 task と重複しない。
★ 08:30 は LiveOrchestrator (SAT/SUN only) + AnomalyCheck (DAILY) 同時 fire するが、 別 process / 別 log で衝突せず。

---

## 5. 5/18 朝 推奨 実施 timing

### ★ 第一候補: 06:30-07:00 (★ 推奨 ★)

| 時刻 | 状態 |
|------|------|
| 06:00 | DailyJrdbKyi 完了済 |
| 06:30 | AnomalyCheck-0630 登録直後に即 fire 可能 |
| **06:30-07:00** | **★ 本作業実施 ★** |
| 08:00 | DailyPredict 開始 (未着手) |
| 08:30 | LiveOrchestrator 初回 fire (5/18 は SUN、 即 fire) |

→ 余裕 1h+、 LiveOrchestrator 初回 fire 前 に確実 登録完了。

### 代替候補: 5/18 当日朝の他 timing

- **05:00-06:00**: 全 task 静かな枠、 余裕 3h。 早起き OK なら最良。
- **07:00-07:50**: DailyPredict 直前、 登録後 08:00 自動 fire まで 10-60 min バッファ。

### 代替候補: 5/18 夜 (★ 朝に余裕ない場合 ★)

- **20:30 (DailyResultsEvening 20:00 完了後)**: admin 入力 1 回で 1 週末分先送り可能。
- 5/24 (SAT) 初回 fire まで余裕、 5/18 (SUN) の paper eval は skip となる (機会損失 1 day)。

---

## 6. 5/24 (SAT) 初回 paper eval 確認手順 (★ 5/18 でも同様 ★)

### 朝 08:30-09:00

1. **08:30** LiveOrchestrator-15min 自動 fire
2. `logs/live_orchestrator.log` (or live_orchestrator_YYYYMMDD.log) で 起動確認
3. `data/recalc_15min/YYYYMMDD/` 出力 file 確認 (race 数分)
4. Discord `#updates` channel で recalc 通知受信確認 (順位変動あり race のみ)
5. **異常時** 即 rollback (Section 8 参照)

### 夜 (DailyResultsEvening 完了後)

- `data/recalc_15min/YYYYMMDD/` で 12-18 R 蓄積確認
- 戦略⑦案 C skip 動作確認 (京都 R / 条件 X 含まれず)
- T6 異常 detection trigger 6/7 動作確認

---

## 7. 5/18-6/15 4 週末 paper eval 期間

| 日 | 曜日 | 開催 |
|----|------|------|
| 5/18 | SUN | 初回 fire 日 (登録当日) |
| 5/24 | SAT | week 2 |
| 5/25 | SUN | week 2 |
| 5/31 | SAT | week 3 |
| 6/1  | SUN | week 3 |
| 6/7  | SAT | week 4 |
| 6/8  | SUN | week 4 |
| 6/14 | SAT | week 5 |
| 6/15 | SUN | week 5 |

= 9 day (★ 5/18 含む ★)、 各 day ~12 R (戦略⑦案 C 適用後) = **~108 R 蓄積目標**
最低 30 R で 採用判定可能 (6/17 N3 checklist 参照)。

---

## 8. 異常時 rollback (1 行)

### 全 schtask 一括解除 (admin)

```powershell
schtasks /Delete /TN "Keiba-LiveOrchestrator-15min" /F
schtasks /Delete /TN "Keiba-FeaturesIntegrityCheck" /F
schtasks /Delete /TN "Keiba-AnomalyCheck-0630" /F
schtasks /Delete /TN "Keiba-AnomalyCheck-0830" /F
schtasks /Delete /TN "Keiba-AnomalyCheck-0940" /F
schtasks /Delete /TN "Keiba-AnomalyCheck-1410" /F
schtasks /Delete /TN "Keiba-AnomalyCheck-1700" /F
schtasks /Delete /TN "Keiba-DailyCumulativeAudit" /F
```

### 単一 task のみ解除

不調 task 名のみ /Delete 実行。 他 task は維持。

### unregister bat (★ 存在すれば こちら推奨 ★)

```powershell
Glob "tools/unregister_*_schtask.bat"
```

存在する unregister bat があれば 順次実行。

---

## 9. 動作確認 checklist (5/18 admin 実行後)

- [ ] `schtasks /Query | findstr Keiba` で 新規 8 件登録確認
- [ ] 各 schtask の Next Run Time が期待 timing
- [ ] `logs/` directory が書き込み可能
- [ ] Discord webhook 環境変数 (`DISCORD_WEBHOOK_UPDATES`) 設定確認
- [ ] V15 production 不変確認 (`git status` で predict_core / daily_predict / app.py / .pkl.gz 全 clean)
- [ ] 翌週末 (5/24) の DailyPredict + LiveOrchestrator 連動確認 (or 5/18 当日 即確認)
- [ ] 5/18 08:30 LiveOrchestrator 初回 fire の log 確認 (★ 登録直後 ★)
- [ ] 5/18 21:00 CumulativeAudit 初回 fire の log 確認 (★ 当日夜 ★)
- [ ] 5/18 22:00 FeaturesIntegrity 初回 fire の log 確認 (★ 当日夜 ★)

---

## 10. ★ G1 day 後の運用 (5/17 → 5/18 移行) ★

- **5/17 (土) G1 day** 終了後の夜は休息、 admin 操作なし
- **5/18 (日) 朝 06:30-07:00** で schtask 登録 (本 doc 手順)
- **5/18 (日) 08:30** LiveOrchestrator 初回 fire
- 5/18 当日 paper eval 開始 → 5/24 / 5/25 / 5/31... と 9 day 継続

---

## 付録 A: 参照ファイル

| ファイル | 用途 |
|---------|------|
| `tools/p0_5_schtask_register.bat` | LiveOrchestrator 登録 |
| `tools/register_features_integrity_schtask.bat` | FeaturesIntegrity 登録 |
| `tools/register_anomaly_detector_schtask.bat` | AnomalyCheck 5 件 登録 |
| `tools/register_daily_cumulative_audit_schtask.bat` | CumulativeAudit 登録 |
| `tools/live_orchestrator.bat` | LiveOrchestrator 本体 (schtask 起動先) |
| `tools/anomaly_auto_detector.bat` | AnomalyCheck 本体 (schtask 起動先) |
| `tools/daily_cumulative_audit.bat` | CumulativeAudit 本体 (schtask 起動先) |
| `tools/features_integrity_monitor.py` | FeaturesIntegrity 本体 (python) |

## 付録 B: 1 行 動作確認 cmd 集

```powershell
# 新規 8 件 一括確認
schtasks /Query /FO LIST | findstr "Keiba-LiveOrchestrator Keiba-FeaturesIntegrity Keiba-AnomalyCheck Keiba-DailyCumulativeAudit"

# Next Run Time 確認
schtasks /Query /FO LIST /V | findstr /C:"TaskName" /C:"Next Run Time" | findstr /B "TaskName" /A:"Next Run"

# 既存全 Keiba task list
schtasks /Query /FO TABLE | findstr "Keiba"
```

---

★ honest 厳守、 V15 完全不変、 admin 操作は本 doc の手順のみ ★
