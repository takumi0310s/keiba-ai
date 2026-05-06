# 5/9 (土) リハーサル計画 — 当日 3 段階予測機構

**作成**: 2026-05-06 朝活 (Session #28 E)
**対象**: 5/9 (土) 本番、5/10 (日) 本番

---

## 1. 全体スケジュール (5/8 金 → 5/9 土)

### 5/8 (金)

| 時刻 | task | 自動/手動 | 備考 |
|------|------|----------|------|
| 10:00 | Keiba-FridayWeekendScrape | 自動 | 5/9 開催情報事前取得 |
| 21:00 | 5/9 開催情報確認 | **手動** | race_name + 12R 1勝クラス候補 |
| 21:00 | refresh_cookie --check | **手動** | 1 分 |
| 21:30 | admin schtasks 4 件 (累計、未済なら) | **手動** | 下記 § 4 参照 |
| 22:00 | 5/3 データで dry-run リハーサル | **手動** | 下記 § 3 参照 |

### 5/9 (土) 自動運用

| 時刻 | task | 内容 | Discord ch |
|------|------|------|-----------|
| 06:30 | Keiba-Morning_Sat | morning_top_races (V17 11R/12R 軸候補) | #bets |
| 08:00 | DailyPredict (watchdog) | 全 R 予測 → daily_predictions/20260509.csv | #bets |
| 08:50 | Keiba-AM8FireCheck | 発火確認 | #updates |
| 09:00 | Keiba-JrdbRetryAm9_Sat | TYB/SED/KYI/KAB retry (commit 95495268) | #updates |
| 09:30 | Keiba-MorningWeightCheck_Sat | 案B改 採用候補のみ早朝補正 | #updates |
| **10:00** | **Keiba-MultiStagePredict_Test10_Sat** | **2R 馬体重補正 + 3R-12R 朝予測通知** | **#updates** |
| **14:50** | **Keiba-MultiStagePredict_Race11_1450_Sat** | **全 11R 予測 (重賞含む)** | **#updates** |
| **15:45** | **Keiba-MultiStagePredict_Race12_1545_Sat** | **全 12R 予測 + 採用 R 買い目 (主戦場)** | **#updates** |
| 14:00-15:30 | PAT 投票 | **ユーザー手動** | (案B改 採用 R × 700 円、上限 2,100 円) |
| 18:00 | DailyResults_Sat + RaceDayReport_Sat | 結果照合 + Discord report | #updates |
| 20:00 | DailyResultsEvening | 結果照合 (二重) | #updates |
| 20:30 | post_5_9_improvement_template.md 振り返り | **ユーザー手動** | 30 min |

---

## 2. 5/8 (金) 21:00 後 確認 (1 度だけ、必須)

```bash
cd C:\Users\takum\keiba-ai

# 5/9 全 race_id + 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}', headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '->', (m.group(1).strip() if m else 'NOT_FOUND'))
"

# Cookie 健全性
python tools/refresh_cookie.py --check

# 朝予測予習 (オフライン dry-run)
# 09:30 Morning_Sat はまだなので、現時点 (5/8 夜) では daily_predictions/20260509.csv 不在
# 5/9 06:30 以降に確認
```

---

## 3. 5/8 (金) 22:00 dry-run リハーサル

```bash
cd C:\Users\takum\keiba-ai

# 5/3 データを使った 3 stage 動作確認
python tools/multi_stage_predict.py --stage test10 --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race11_1450 --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race12_1545 --date 20260503 --dry-run
```

期待:
- 各 stage で predict_one_race が 3 R 成功
- Discord format が想定通り (--dry-run で送信なし)
- CSV 保存 OK

→ 5/3 で OK なら 5/9 本番でも OK。 `data/v18/multi_stage_predict_test_5_6.md` 参照。

---

## 4. 5/8 (金) までに admin で 1 度実行 (累計 4 件、未済分のみ)

```powershell
# 1. ProcessWatchdog v2 (commit 86cd1da5)
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_process_watchdog_v2.ps1

# 2. 馬体重補正 (commit 7358a74a、09:30 早朝)
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_morning_weight_check_schtasks.ps1

# 3. JRDB AM 9:00 retry (commit 95495268)
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_jrdb_retry_schtasks.ps1

# 4. multi_stage_predict (Session #28、本書、10:00/14:50/15:45)
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_multi_stage_predict_schtasks.ps1
```

→ admin PowerShell で各 1 コマンド、合計 4 コマンド実行で 5/9 自動運用準備完了。

確認:
```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-*' -or $_.TaskName -eq 'ProcessWatchdog' } | ft TaskName, State -AutoSize
```

期待: 全 task State=Ready、ProcessWatchdog のみ Ready (Disabled から)。

---

## 5. 5/9 (土) 朝の フロー (時系列)

### 6:30 - 9:00 自動発火 (USER は寝てる/起きてる、操作不要)

- 06:30 Morning_Sat → Discord #bets で 11R/12R 軸候補通知
- 08:00 DailyPredict → daily_predictions/20260509.csv 生成
- 08:50 AM8FireCheck → "OK" 通知
- 09:00 JrdbRetryAm9 → TYB/SED retry
- 09:30 MorningWeightCheck → 案B改 採用候補 早朝補正

### 9:30 - 10:00 USER 手動確認 (5 分)

- Discord #updates で 09:30 MorningWeightCheck 通知確認
- `data/results/20260509_pat_checklist.md` の確認 (5/9 朝必読)
- 12R race_name が 1 勝クラスかどうか目視

### 10:00 - 14:50 自動 + 待機

- **10:00 MultiStagePredict_Test10** ← 機構動作確認 (Discord #updates)
   - 2R 馬体重補正 + 3R-12R 朝予測通知
   - **これを見て機構が正常か確認**、異常なら手動介入
- 14:50 まで USER は待機 (買い物/食事/別作業)

### 14:50 11R 一括予測

- **14:50 MultiStagePredict_Race11_1450** ← Discord #updates
   - 全 3場 11R 予測 (重賞含む)
   - 案B改 では基本 採用外 (重賞 + OP/特別)
   - 観察用 (5/16 V18/V19 試行の前提データ)

### 15:25 - 15:45 11R 発走 + 結果

- 15:25 11R 発走、結果は 15:45 までに確定

### 15:45 12R 一括予測 ★主戦場

- **15:45 MultiStagePredict_Race12_1545** ← Discord #updates
   - 全 12R 予測 + 採用 R の買い目
   - 案B改 1勝クラスのみ採用、買い目 三連複 7 点 700 円
   - **これを見て PAT 投票 (14:50 時点でも判断は可能だが、馬体重最新確認)**

### 15:45 - 16:20 PAT 投票

- 採用 R × 700 円 (上限 2,100 円 = 3 R)
- 買い目は Discord 通知の trio_bets 列をそのまま入力

### 16:20 - 17:00 12R 発走 + 結果

- 12R 発走、結果は 16:50 頃までに確定

### 18:00 結果照合

- DailyResults_Sat + RaceDayReport_Sat → Discord #updates
- 累計収支 表示 (cumulative_results.csv 更新)

### 20:30 振り返り

- `data/v18/post_5_9_improvement_template.md` 埋め (30 min)
- 5/16 戦略決定の input

---

## 6. 想定 alert / トラブル 対応

| 症状 | 対応 |
|------|------|
| 10:00 Test10 で予測失敗 | logs/multi_stage_predict_test10_20260509.log 確認、Cookie 切れなら refresh_cookie --auto |
| 14:50 Race11_1450 で全 R 採用外 | 想定通り (案B改 で 11R は基本採用外)、Discord 表示確認のみ |
| 15:45 Race12_1545 で採用 R 0 件 | 12R 全部が 1 勝以外 → 投資なし、5/9 撤退ライン余裕維持 |
| Discord 通知来ない | DISCORD_WEBHOOK_URL 確認 + notify_done.py で test |
| schtasks 発火しない | Get-ScheduledTask + Start-ScheduledTask で手動発火 |

---

## 7. 5/9 採用 R 想定 (5/8 21:00 後 確定)

| 開催 | 12R | 採用候補? |
|------|-----|----------|
| 東京 12R | (5/8 21:00 後 race_name 確認) | 1勝クラスなら採用 |
| 京都 12R | 同上 | 1勝クラスなら採用 |
| 新潟 12R | 同上 | 1勝クラスなら採用 |

過去傾向 (4/26-5/3 cumulative_results 集計):
- 12R は 1勝/2勝/未勝利/特別 が ばらつき
- 案B改 採用は 1勝のみ、平均 0-3 R/日

5/9 想定: **0-3 R 採用、投資 0-2,100 円、最悪 -2,100 円、累計余裕 +61,430 円維持**。

---

## 8. リハーサル成功基準

| 項目 | 基準 |
|------|------|
| 10:00 Test10 | 2R 全 3場 予測成功、Discord OK |
| 14:50 Race11_1450 | 11R 全 3場 予測成功、採用判定正確 |
| 15:45 Race12_1545 | 12R 全 3場 予測成功、案B改 採用判定正確 |
| 案B改 投票 | 採用 R × 700 円、上限 2,100 円遵守 |
| 撤退ライン | -50,000 円超え NG |

→ 5/8 22:00 dry-run + 5/9 本番自動発火 で全項目 PASS なら **5/16 試行 (V18/V19/NAR) へ移行可**。

---

## 9. 結論

5/9 (土) 当日 3 段階予測機構が完成、4 つの admin schtasks (累計) で全自動化準備完了。 USER 手動操作は:
- 5/8 (金) 21:00 後 race_name 確認 (5 min)
- 5/8 (金) 22:00 dry-run リハーサル (10 min)
- 5/8 までに admin schtasks 4 件 (累計、未済分のみ、各 1 分)
- 5/9 (土) 14:00-15:30 PAT 投票 (15 min)
- 5/9 (土) 20:30 振り返り (30 min)

合計 60 分程度で 5/9 完遂。 残りは自動化済。
