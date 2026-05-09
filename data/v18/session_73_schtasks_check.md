# Session #73 C: schtasks 5/10 fire 確認

実行日時: 2026-05-09 18:30+

## Keiba 系 task 総数

`52 件` (prompt 想定 50 件 から +2)。
増分は 5/9 系 verdict / cumulative / summary task と思われる。

## 5/10 (Sun) 朝 fire 順 (時系列)

| 時刻 | task | 状態 | 備考 |
|------|------|------|------|
| 06:30 | `\Keiba-Morning_Sun` | Ready | 朝 dashboard |
| 07:00 | `\Keiba-MorningDigest` | Ready | 毎日 |
| 07:30 | `\keiba-ai\JrdbHealthCheck_Sun` | Ready | JRDB 取得 健全性 check |
| 08:00 | `\keiba-ai\DailyPredict` | Ready | ★ 当日全 R 朝予測 ★ |
| 08:45 | `\keiba-ai\RaceAutoNotify_Sun` | Ready | top3 通知 (既存) |
| 09:30 | `\Keiba-MorningWeightCheck_Sun` | Ready | 馬体重 alert |
| **09:30** | `\Keiba-SaveAllHorseScores_0930` | **Ready** | **★ Session #71 ★ daily_predictions_full 生成** |
| 10:00 | `\Keiba-MultiStagePredict_Test10_Sun` | Ready | multi stage test |

## 5/10 全日 (午後 + 夜)

| 時刻 | task | 状態 |
|------|------|------|
| 14:50 | `\Keiba-MultiStagePredict_Race11_1450_Sun` | Ready |
| 15:45 | `\Keiba-MultiStagePredict_Race12_1545_Sun` | Ready |
| 18:00 | `\Keiba-RaceDayReport_Sun` | Ready |
| 18:00 | `\keiba-ai\DailyResults_Sun` | Ready |

## 5/9 限定 task (5/10 silent fail or no-op)

| task | 5/10 動作 | 影響 |
|------|----------|------|
| `\Keiba-PreRacePredict_Watchdog_5_9` | **silent fail** | Session #72 機能 完全停止 (★ B doc 参照 ★) |
| `\Keiba-Cumulative_1700_5_9` | 5/9 限定 fire | 5/10 fire しない |
| `\Keiba-Summary_2030_5_9` | 5/9 限定 fire | 5/10 fire しない |
| `\Keiba-VoteCandidates_1400_5_9` | 5/9 限定 fire | 5/10 fire しない |
| `\Keiba-Verdict_R11_*` (3件) | 5/9 限定 | 5/10 fire しない |
| `\Keiba-Verdict_R12_*` (3件) | 5/9 限定 | 5/10 fire しない |

## PreRacePredict_Watchdog_5_9 詳細

```
Schedule: One Time Only, Minute (Repeat 30 min, Duration 700h)
Start Date: 2026/05/09 13:00
Last Run: 2026/05/09 19:30:01 (Result 0、 但し bat 不在で空 exit の可能性)
Next Run: 2026/05/09 20:00
Task To Run: wscript.exe ... pre_race_predict_runner.bat --check-next-1h
```

問題:
1. `pre_race_predict_runner.bat` は dev/two-stage commit、 main 不在
2. 現在 main checkout 中 → bat 物理欠如 → silent_runner.vbs 空 exit
3. stage2_predict.py も 5/9 hardcode (★ B doc 参照 ★)

→ 5/10 朝 PreRacePredict_Watchdog_5_9 は fire しても silent fail。

## SaveAllHorseScores_0930 詳細

```
Schedule: Weekly SUN, SAT
Start Date: 2026/05/09 09:30
Last Run: 1999/11/30 (= 未 fire)
Next Run: 2026/05/10 09:30
Task To Run: wscript.exe ... save_all_horse_scores_runner.bat
Last Result: 267011 (= ERROR_PATH_NOT_FOUND の可能性、 但し未 fire のため意味薄)
```

問題なし: 5/10 9:30 fire 時点で:
- DailyPredict 8:00 完了 → daily_predictions/20260510.csv 生成済 想定
- save_all_horse_scores_runner.bat main 在中 (Session #71 commit `5f5c3d43`)
- Test (Session #73 A) で graceful behavior 確認済

## 整合性 判定

| 項目 | 状態 |
|------|------|
| ★ DailyPredict 8:00 ★ | OK ✓ |
| ★ SaveAllHorseScores 9:30 ★ | OK ✓ (Session #71 動作見込み) |
| RaceAutoNotify_Sun 8:45 | OK ✓ |
| MorningWeightCheck_Sun 9:30 | OK ✓ |
| MorningDigest 7:00 | OK ✓ |
| Morning_Sun 6:30 | OK ✓ |
| **PreRacePredict_Watchdog_5_9** | **★ silent fail (5/10 動作不能) ★** |
| Verdict_R*_5_9 系 | 5/10 fire しない (意図通り) |

## V15 投資保護

5/10 朝の主要 task (DailyPredict / RaceAutoNotify / SaveAllHorseScores) は全 OK。
PreRacePredict_Watchdog の silent fail は補助 通知の停止のみで、
V15 production / 投票 logic / 累計収支 +13,530 円 に影響なし。

## 次 action

→ Session #73 D (runbook) で 5/10 朝 fire 失敗時の手順 doc 化。
