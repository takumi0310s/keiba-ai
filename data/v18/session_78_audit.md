# Session #78 audit — stage2_predict.py 5/9 hardcode

日時: 2026-05-09 19:30 過ぎ
契機: 5/10 朝 silent fail 懸念 → 寝る前に root cause 解決

## 現状把握

| 項目 | 値 |
|------|----|
| schtask | `Keiba-PreRacePredict_Watchdog_5_9` (Last Result: 0、 30 分間隔、 13:00-700h 反復) |
| 起動 cmd | `wscript.exe silent_runner.vbs pre_race_predict_runner.bat --check-next-1h` |
| runner stub (Session #77) | `if exist tools\stage2_predict.py` で no-op (main 上で missing → 黙って exit 0) |
| 本体所在 | `dev/two-stage` のみ (main には無い) |
| 本体 LoC | 557 行 |

## hardcode 箇所 (dev/two-stage HEAD)

| line | 内容 | 影響 |
|------|------|------|
| 41 | `DATE = "20260509"` | 5/10 以降 不在 csv を読みに行く → 即 fail |
| 45 | `CACHE_PATH = OUT_DIR / "pre_race_predict_cache_5_9.json"` | dedup cache が 永久に 5/9 用 |
| 460 | `out_path = OUT_DIR / f"pre_race_predict_5_9_R..."` | 出力 json の prefix が 5/9 固定 |

## silent fail 経路

1. schtask 30 分毎 fire
2. silent_runner.vbs → pre_race_predict_runner.bat (Session #77 stub)
3. main 上で `tools\stage2_predict.py` 不在 → no-op exit 0
4. ★ 状態的には fail していないが、 機能としては Stage 2 予測が完全 silent skip ★
5. 5/10+ にこの状態が継続 → 補助通知ゼロ (V15 朝予測には影響なし)

## 5/10 朝 影響

- ✓ DailyPredict_0800 (V15 案B改 strict): **影響なし**
- ✓ SaveAllHorseScores_0930 (Session #71): **影響なし**
- ✓ RaceAutoNotify_Sun 0845: **影響なし**
- ⚠ PreRacePredict_Watchdog_5_9: 黙って no-op (補助通知のみ消失)

## 結論

V15 投資保護には影響なし。 ただし 5/15 V18 trial で Stage 2 通知を活用予定。
方針: schtask DISABLE + dev/two-stage 側で hardcode 撤廃 (両方やる、 Session #78 提案 ★推奨★ 通り)。
