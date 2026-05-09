# Session #73 A: 5/10 manual dry run 結果

実行日時: 2026-05-09 18:30+
対象: tools/save_all_horse_scores.py (Session #71 実装)

## 5/10 開催 (Sun)

5/10 (日) 中央 3 場開催想定 (5/9 と同じ 京都/東京/新潟):
- 京都 8 開催 6 日目 (race_id prefix 202608030 6XX)
- 東京 5 開催 6 日目 (race_id prefix 202605020 6XX)
- 新潟 4 開催 1 開催 4 日目 (race_id prefix 202604010 4XX)

5/10 race_ids は daily_predict.py 8:00 fire 後に確定。

## Test 1: 5/10 dry-run (csv 未生成 case)

```
python tools/save_all_horse_scores.py --date 20260510 --dry-run
```

結果:
```
[STEP 0] V15 model load...
[MODEL] v15 Pattern B (当日情報込み, 150特徴量) ロード成功 (keiba_model_v15_central_live.pkl.gz)
  loaded: is_live=True
[INFO] 20260510 の race 一覧見つからず (daily_predict.py 未完了 or 非開催)
```

判定: PASS
- V15 Pattern B model load 動作 OK
- daily_predictions/20260510.csv 不在時の graceful exit 動作 OK
- 5/10 朝 8:00 daily_predict.py 完了後、 9:30 SaveAllHorseScores fire で正常実行見込み

## Test 2: 5/9 1 R inference (model + flow 検証)

```
python tools/save_all_horse_scores.py --date 20260509 --race-id 202608030501 --dry-run
```

結果:
```
[STEP 1] 対象 R: 1
[1/1] race_id=202608030501
  [SKIP] 馬データなし
[STEP 2] inference 完了 (0.3s, 1 R, 0 row, 1 fail)
```

判定: PARTIAL PASS (期待動作)
- 5/9 既終了 R に対して parse_shutuba は出馬表 取得 不可 (仕様、 終了後は db.netkeiba 結果 page に置換)
- 但し model load + csv read + race loop + skip handling は全て期待通り
- 5/10 朝 9:30 fire 時点では当日 R の shutuba 利用可能 (発走前) なので 全頭 inference 動作見込み

## 5/10 朝 fire 順 想定

| 時刻 | task | 期待動作 |
|------|------|---------|
| 06:30 | morning_checklist | 当日 fire 予定 task 全 check + Discord |
| 08:00 | DailyPredict_0800 | data/daily_predictions/20260510.csv 生成 |
| 08:45 | RaceAutoNotify_0845 | top3 通知 (既存) |
| 09:00 | PreRacePredict_Watchdog | 1h 前 全馬通知 (Session #65 + #72) |
| 09:30 | SaveAllHorseScores_0930 | data/daily_predictions_full/20260510.csv 生成 (★ Session #71 ★) |

## 残課題

なし。 5/10 朝 9:30 fire 時点で daily_predictions/20260510.csv が生成済 (8:00 fire) なので、
SaveAllHorseScores 正常動作 見込み。

## 失敗 case

[docs/RUNBOOK_5_10_DRY_RUN.md](../../docs/RUNBOOK_5_10_DRY_RUN.md) 参照 (Session #73 D)。
