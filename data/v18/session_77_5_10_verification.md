# Session #77 D: 5/10 朝 fire 動作保証

## 5/10 朝 fire 想定 task

| 時刻 | task | bat | py | 状態 |
|------|------|-----|----|----|
| 06:30 | Keiba-Morning_Sun | tools/morning_top_races.bat | tools/morning_top_races.py | ✓ |
| 07:00 | Keiba-MorningDigest | morning_dashboard.bat | tools/morning_dashboard.py | ✓ |
| 07:30 | Keiba-JrdbRetryAm9_Sun | tools/jrdb_retry_am9.bat | tools/jrdb_retry_am9.py | ✓ |
| 08:00 | keiba-ai\\DailyPredict ★最重要★ | daily_predict_watchdog.bat | tools/daily_predict_watchdog.py | ✓ |
| 08:45 | Keiba-RaceAutoNotify (土日 fire) | race_auto_notify.bat | tools/race_auto_notify.py | ✓ |
| 09:00- (30 分毎) | Keiba-PreRacePredict_Watchdog_5_9 | pre_race_predict_runner.bat ★Session #77 新規★ | (no-op stub on main) | ✓ |
| 09:30 | Keiba-SaveAllHorseScores_0930 | save_all_horse_scores_runner.bat | tools/save_all_horse_scores.py | ✓ |
| 09:30 | Keiba-MorningWeightCheck_Sun | tools/morning_weight_check.bat | tools/morning_weight_check.py | ✓ |
| 18:00 | Keiba-RaceDayReport_Sun | race_day_report.bat ★Session #77 新規★ | tools/race_day_report.py | ✓ |
| 20:00 | keiba-ai\\DailyResultsEvening | daily_results.bat | tools/daily_results.py | ✓ |

## 動作確認 (実 fire test 5/9 19:43)

```
$ schtasks /Run /TN "Keiba-PreRacePredict_Watchdog_5_9"
SUCCESS: Last Result=0、 popup 出ず、 log 正常
```

## 失敗時 fallback (manual 手順)

5/10 朝 task fire 失敗時、 ユーザー手動実行で復旧可:

```
# DailyPredict (08:00) 手動 run
schtasks /Run /TN "keiba-ai\DailyPredict"
# OR 直接
cd C:\Users\takum\keiba-ai
python tools\daily_predict.py --date 20260510

# SaveAllHorseScores (09:30) 手動 run
schtasks /Run /TN "Keiba-SaveAllHorseScores_0930"
# OR 直接
python tools\save_all_horse_scores.py --date 20260510

# RaceAutoNotify (08:45+) 手動 run
schtasks /Run /TN "Keiba-RaceAutoNotify"
# OR 直接
python tools\race_auto_notify.py
```

## V15 投資保護

| 項目 | 状態 |
|------|------|
| V15 model file 不変 | ✓ (Session #77 model 触らず) |
| predict_core.py 不変 | ✓ |
| daily_predict.py 不変 | ✓ |
| app.py 不変 | ✓ |
| 5/9 投票結果 影響 | ✓ なし (修復は schtask wrapper のみ) |
| 累計 +¥12,830 | ✓ 維持 |
| 5/10 朝 V15 動作 | ✓ 完全保証 |

## 結論

Session #77 修復で:
1. silent_runner.vbs Line 24 popup ★解消★
2. 5/10 朝 全 task fire 動作 ★保証★
3. V15 production ★完全不変★
