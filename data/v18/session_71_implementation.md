# Session #71 B+C: tools/save_all_horse_scores.py 実装

## CLI
```
python tools/save_all_horse_scores.py --date 20260510            # 本番
python tools/save_all_horse_scores.py --date today               # 今日
python tools/save_all_horse_scores.py --date 20260510 --dry-run  # csv 保存なし
python tools/save_all_horse_scores.py --date 20260510 --race-id 202604010312  # 単一 R test
```

## logic (要点)

1. **kill-switch 最優先**: `data/v18/save_all_horse_scores.kill` 存在で即 no-op exit (Session #64 思想)
2. **model 1 回 load** (`load_models()`): 36 R で再 load しない overhead 削減
3. **race_id list**: `get_race_ids_from_existing_csv(date_str)` で `daily_predictions/{date}.csv` を read-only で読み込み (retry 1 回)
4. **各 R inference**: `predict_one_race_full()` で daily_predict.py の race loop と完全同一 flow
   - parse_shutuba → odds (real-time or result) → JRA 馬場・天候 (cache) → get_horse_stats × N → build_features → merge_jrdb_predict_features → predict_race
5. **全頭 sort + rank**: `sorted_df = df.sort_values('スコア', ascending=False)`、 `rank_in_race` 1〜N
6. **CSV write**: `data/daily_predictions_full/{date}.csv` に 全 row 書き出し (append でなく overwrite、 1 file = 1 day)

## CSV schema

```
date, race_id, course, race_num, race_name, num_horses, distance, surface,
condition, horse_num, horse_name, horse_id, V15_score, rank_in_race,
popularity, odds, source
```

`source = "v15_production_full"` を全 row に明記 (Session #70 LEAK 防止思想)。

## 並行実行 安全性

- daily_predict.py と読み書き範囲が完全分離 (in: daily_predictions/、 out: daily_predictions_full/)
- model file は read-only で複数 process が同時 load OK
- locking 不要

## error handling

- model load 失敗 → log + sys.exit(1)
- csv read 失敗 → 30s sleep + retry 1 回
- 個別 R inference 失敗 (parse_shutuba fail / 障害除外) → log + skip + 部分保存
- 全 R fail → log + sys.exit(2)

## 動作確認 (5/9 18:01 dry-run)

| test | 結果 |
|---|---|
| import (`predict_core`, `jrdb_features`) | OK |
| kill-switch (`.kill` file 配置) | 即 no-op exit OK |
| `--date 20260510` (csv 不在) | 「race 一覧見つからず」 clean exit OK |
| `--race-id 202604010312` (過去 R) | parse_shutuba が「馬データなし」 (過去 R は shutuba page 期限切れ、 想定内) |

★ 過去 R inference は意味がない (LEAK risk + page 構造変化)。 本来 5/10 8:30+ daily_predict 完了後の **未来 R** に対して走る ★

## 既存 file 不変 verification

```
$ git status -s tools/predict_core.py tools/daily_predict.py app.py
(空) — 全 不変 ✓

$ md5sum keiba_model_v15_central*.pkl.gz
309dffc65504f056d233c65665c319d5 *keiba_model_v15_central.pkl.gz
fac1588ac20e96ae81eef1efbf7f423e *keiba_model_v15_central_live.pkl.gz
↑ V15 model 不変 ✓
```
