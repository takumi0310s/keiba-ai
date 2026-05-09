# Session #71 D: schtask 追加

## 新 schtask: `Keiba-SaveAllHorseScores_0930`

| 項目 | 値 |
|---|---|
| 名前 | `Keiba-SaveAllHorseScores_0930` |
| Schedule | `/SC WEEKLY /D SAT,SUN /ST 09:30` (週末 9:30、 DailyPredict 完了見込み後) |
| Action | `wscript.exe silent_runner.vbs save_all_horse_scores_runner.bat` |
| log | `logs/save_all_horse_scores.log` |
| Admin | 不要 |

## 時刻決定の根拠

- 既存 `\keiba-ai\DailyPredict` は 8:00 fire、 5/9 実績で 8:00-8:56 (56 分) 掛かった
- 元案 8:30 → daily_predict 未完了で `daily_predictions/{date}.csv` 不在 or 部分のみ → 新 tool が race 一覧取れない or partial run
- 8:45 RaceAutoNotify_Sun と netkeiba rate-limit 競合の懸念
- → **9:30 に調整** (DailyPredict 9:00 までに完了見込み + RaceAutoNotify quick scan 完了後)
- 9:30 `MorningWeightCheck_Sun` と並走するが、 別 endpoint なので OK

## 既存 schtasks 49 件 不変、 **+1 件のみ**

```
\Keiba-SaveAllHorseScores_0930  2026/05/10 09:30:00  Ready
```

## kill-switch 互換

`data/v18/save_all_horse_scores.kill` を touch すれば次 fire 以降 即 no-op exit (Session #64 思想)。

## 期待 wall time

- 36 R × ~30s/R (parse_shutuba + odds + horse_stats×N + JRDB merge + predict) = **~18 分**
- 9:30 開始 → 9:48 完了見込み
- 13:00 NarMidDayCalendar / 14:50 MultiStagePredict_Race11 までは余裕

## 5/10 朝 動作シナリオ

```
06:00 DailyJrdbKyi
06:30 Morning_Sun
07:00 MorningDigest
07:30 JrdbHealthCheck_Sun
08:00 DailyPredict (~8:56 完了)
08:45 RaceAutoNotify_Sun
09:00 JrdbRetryAm9_Sun
09:30 ★ Keiba-SaveAllHorseScores_0930 ★ (新規) — 完了 ~9:48
09:30 MorningWeightCheck_Sun (並走 OK)
10:00 MultiStagePredict_Test10_Sun
...
```
