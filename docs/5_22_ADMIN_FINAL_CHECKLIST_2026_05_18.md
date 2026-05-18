# 5/22 PM admin final checklist (★ 寝起き 1 doc で 5/23 fire ready ★)

## ★ 0. 開始 timing ★
- 5/22 (金) 21:00 開始推奨 (5/23 (土) 08:30 LiveOrchestrator 初回 fire まで 11.5h 余裕)
- 所要時間: 20-30 分
- 必要: admin 権限 PowerShell

## 1. PowerShell 起動
- Win + X → "Windows PowerShell (管理者)" 選択
- cd C:\Users\takum\keiba-ai

## 2. 現状確認 (★ 0/9 未登録 が想定 ★)
- python tools/admin_verify_v2.py
- 出力 JSON で:
  - schtasks_registered = "0/9"
  - bats_exist = "9/9"  (完成-1 完了後)
- もし bats_exist < 9/9 → 完成-1 未完、 admin 操作 abort

## 3. 9 schtask /Create 順次実行 (★ admin cmd ★)

### 3.1 LiveOrchestrator-15min (土日 08:30-17:00 polling)
```powershell
schtasks /Create /TN "Keiba-LiveOrchestrator-15min" /TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat" /SC WEEKLY /D SAT,SUN /ST 08:30 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-LiveOrchestrator-15min"  # exit code 0 確認
```

### 3.2 FeaturesIntegrity-Daily (daily 22:00)
```powershell
schtasks /Create /TN "Keiba-FeaturesIntegrity-Daily" /TR "C:\Users\takum\keiba-ai\tools\keiba_features_integrity.bat" /SC DAILY /ST 22:00 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-FeaturesIntegrity-Daily"
```

### 3.3-3.7 AnomalyCheck (5 件、 土日のみ各 time)
```powershell
# 0630
schtasks /Create /TN "Keiba-AnomalyCheck-0630" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0630.bat" /SC WEEKLY /D SAT,SUN /ST 06:30 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-AnomalyCheck-0630"

# 0830
schtasks /Create /TN "Keiba-AnomalyCheck-0830" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0830.bat" /SC WEEKLY /D SAT,SUN /ST 08:30 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-AnomalyCheck-0830"

# 0940
schtasks /Create /TN "Keiba-AnomalyCheck-0940" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0940.bat" /SC WEEKLY /D SAT,SUN /ST 09:40 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-AnomalyCheck-0940"

# 1410
schtasks /Create /TN "Keiba-AnomalyCheck-1410" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1410.bat" /SC WEEKLY /D SAT,SUN /ST 14:10 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-AnomalyCheck-1410"

# 1700
schtasks /Create /TN "Keiba-AnomalyCheck-1700" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1700.bat" /SC WEEKLY /D SAT,SUN /ST 17:00 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-AnomalyCheck-1700"
```

### 3.8 CumulativeAudit-Daily (daily 21:00)
```powershell
schtasks /Create /TN "Keiba-CumulativeAudit-Daily" /TR "C:\Users\takum\keiba-ai\tools\keiba_cumulative_audit.bat" /SC DAILY /ST 21:00 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-CumulativeAudit-Daily"
```

### 3.9 RaceNotifyLogV2Aggregator (土日 20:30)
```powershell
schtasks /Create /TN "Keiba-RaceNotifyLogV2Aggregator" /TR "C:\Users\takum\keiba-ai\tools\keiba_race_notify_log_v2_aggregator.bat" /SC WEEKLY /D SAT,SUN /ST 20:30 /RU "SYSTEM" /F
schtasks /Query /FO LIST /TN "Keiba-RaceNotifyLogV2Aggregator"
```

## 4. 全 9 件登録完了 verify
- python tools/admin_verify_v2.py
- 期待: schtasks_registered = "9/9"
- もし < 9/9 → 該当 schtask の /Query で error 確認、 3 の cmd を re-run

## 5. 異常時 rollback
- 個別 delete: schtasks /Delete /TN "Keiba-XXX" /F
- 全 9 件 delete:
  ```powershell
  foreach ($n in @("Keiba-LiveOrchestrator-15min","Keiba-FeaturesIntegrity-Daily","Keiba-AnomalyCheck-0630","Keiba-AnomalyCheck-0830","Keiba-AnomalyCheck-0940","Keiba-AnomalyCheck-1410","Keiba-AnomalyCheck-1700","Keiba-CumulativeAudit-Daily","Keiba-RaceNotifyLogV2Aggregator")) { schtasks /Delete /TN $n /F }
  ```

## 6. ★ V15 production 完全継続 ★
- 5/22 PM admin 操作中も既存 schtask (DailyPredict / RaceAutoNotify / DailyResults / WeeklyReport etc.) は完全継続
- 新 schtask 登録は V15 と並走 (V15 .pkl.gz / predict_core 不変)
- 5/23 (土) 朝 通常 V15 #買い目 通知 + 新 #updates 通知 両方発火想定

## 7. 5/23 (土) SAT 朝 起床後 step
- 06:00 起床 (前夜 normal 就寝想定)
- 06:30 AnomalyCheck-0630 fire 確認 (logs/keiba_anomaly_check_0630_*.log)
- 08:00 既存 DailyPredict fire 確認
- 08:30 LiveOrchestrator-15min 初回 fire 確認 (★ 新規 ★)
- 09:00+ Discord #updates 通知確認 (★ 新規 ★)
- 09:30+ 通常 V15 #買い目 通知確認 (★ 既存 ★)
- 全日 race -15min ごと LiveOrchestrator 連続 fire 想定

## 8. 失敗 path (Plan B)
- 5/23 朝 LiveOrchestrator 不発火 → 既存 V15 + 戦略⑦案 C のみ運用継続 (損失なし、 新規 feature 解禁延期)
- 5/24 (日) 朝までに root cause 究明、 5/25-5/29 で 再 admin 操作

## 9. 5/24 fire ready 確認 final
- 5/23 (土) 21:00 → 5/24 (日) fire 準備
- CumulativeAudit-Daily 21:00 自動 fire (5/23 結果集計)
- 5/24 朝 06:30+ 同 pattern で fire

## 10. ★ honest 注記 ★
- 9/9 schtask 登録 ≠ 即 paper eval / production 投入
- live_data_fetcher は 5/24+ mock のみ (real fetch 5/24+ user 判断 / G1 day blocklist 維持)
- v15_full candidate は paper shadow eval、 V15 .pkl.gz 不変 (6/17 判定後 別 sub-task で予算化)
- Discord 実発火 0 を維持 (mock --dry-run)
