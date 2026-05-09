# Session #78 fix — schtask disable + dev/two-stage hardcode 撤廃

日時: 2026-05-09 19:30 過ぎ
完了: 緊急対応 全項目 OK

## A. schtask disable (main 側、 即時)

```
schtasks /Change /TN "Keiba-PreRacePredict_Watchdog_5_9" /DISABLE
```

確認:
- Status: Disabled
- Scheduled Task State: Disabled

→ 5/10 朝 30 分毎 silent fire が完全停止。 補助通知だけ消失、 V15 production には影響なし。

## B. dev/two-stage 側 hardcode 撤廃

commit: `ae81ebf0` (dev/two-stage、 push 済)

### 変更内容 (`tools/stage2_predict.py`)

| 旧 | 新 |
|----|----|
| `DATE = "20260509"` | `DATE = datetime.now().strftime("%Y%m%d")` (default = 当日) |
| `CACHE_PATH = ... "pre_race_predict_cache_5_9.json"` | `_date_short(DATE)` で動的 (5/9 → "5_9"、 5/10 → "5_10") |
| `out_path = ... f"pre_race_predict_5_9_R..."` | 同じく `_date_short(DATE)` 連動 |
| (CLI args) | `--date YYYYMMDD` / `--dry-run` を追加 |

### test 結果

```
$ python tools/stage2_predict.py --date 20260510 --dry-run
[date] override DATE=20260510
[dry-run] DATE=20260510
[dry-run] DAILY_PRED=...\daily_predictions\20260510.csv exists=False
[dry-run] CACHE_PATH=...\v18\pre_race_predict_cache_5_10.json
[dry-run] KILL_SWITCH=...\v18\pre_race_predict.kill exists=False
[dry-run] OK

$ python tools/stage2_predict.py --dry-run    # 当日 default
[dry-run] DATE=20260509
[dry-run] DAILY_PRED=...\daily_predictions\20260509.csv exists=True
[dry-run] CACHE_PATH=...\v18\pre_race_predict_cache_5_9.json
[dry-run] OK
```

互換性: 既存の `pre_race_predict_cache_5_9.json` は 5/9 default で引き続き読み書き可。

## C. 復旧手順 (5/15 V18 trial 直前)

1. dev/two-stage を main にマージ可否を Session #75 archive plan で再評価
2. それまでは `tools/stage2_predict.py` 本体は dev/two-stage 専用
3. schtask 再有効化:
   ```
   schtasks /Change /TN "Keiba-PreRacePredict_Watchdog_5_9" /ENABLE
   ```
4. 必要なら schtask 名から `_5_9` suffix 撤去 (新 schtask `Keiba-PreRacePredict_Watchdog`)

## D. 状態 (5/10 朝 動作保証)

| schtask | 状態 |
|---------|------|
| ✅ DailyPredict_0800 | V15 案B改 strict (不変) |
| ✅ SaveAllHorseScores_0930 | Session #71 (不変) |
| ✅ RaceAutoNotify_Sun 0845 | 不変 |
| ⏸ PreRacePredict_Watchdog_5_9 | **DISABLED** (5/15 までに re-enable) |

累計 +12,830 円 維持。 V15 投資保護 完全。
