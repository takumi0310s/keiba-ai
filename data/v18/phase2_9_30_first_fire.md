# Phase 2 9:30 SaveAllHorseScores 初稼働 確認 (5/10 09:44)

## 結論: ✅ **動作中** (17/35 R, 48%, ETA 9:58)

## schtasks
```
\Keiba-SaveAllHorseScores_0930
  Last Run:  2026/05/10 9:30:00
  Last Code: 267009 (=TASK_STATUS_RUNNING_INSTANCE)
  Status:    Running
  Next Run:  2026/05/16 9:30:00 (来週 土曜 fire)
```

## Process
```
python pid 37536
  StartTime:  2026/05/10 9:30:00
  CPU sec:    703 (14 min 経過)
  WorkingSet: 1.8 GB
  cmdline:    tools/save_all_horse_scores.py --date 20260510
```

## 進捗 (9:44 時点)
- **17 / 35 R 完了** (48%)
- 京都 1-12R 完了
- 新潟 1, 2, 3, 5 完了
- 残り 18R: 新潟 6-12 + 東京 1-12

各 R の処理:
- ~30 sec/R (JRDB SED+JO+PACI fetch + 馬名フォールバック + V15 inference)
- 残 18R × 30 sec = 9 min → ETA 9:58

## 完了後 artifact
- `data/daily_predictions_full/20260510.csv` (まだ存在せず、 完了時に書き出し)
- 期待 row 数: 約 360-540 (35R × 平均 12-15 頭)
- column: race_id, course, race_num, race_name, condition, num_horses, distance, surface, track_condition, top1_num, top1_name, top1_score, ...
- source = "v15_production_full" (LEAK 防止 mark)

## Session #71 158h+ マラソン集大成
- ✅ kill-switch (`data/v18/save_all_horse_scores.kill`) 機構完備
- ✅ daily_predict.py / race_auto_notify.py 一切 trigger しない (subprocess も NG)
- ✅ ワンショット exit (loop 不可)
- ✅ predict_core 関数 import で同一 logic 再 inference
- ✅ Session #71 から 5/10 まで 158h+ かけた本番 task の **真の初稼働**

## 監視
- 9:58 頃 完了確認: `data/daily_predictions_full/20260510.csv` 存在 + row 数 妥当
- 完了後 LEAK チェック (任意): full csv top1 vs daily_predictions/20260510.csv top1 一致確認

## Discord 通知
- save_all_horse_scores.py 内部から完了通知 出すか不明 (要確認)
- ない場合は cache 9:31:14 / 9:41:22 entries は 別 task (watchdog 等?)
