# Session #71 D: test 結果

## test 環境
- 5/9 18:01
- branch: dev/two-stage (commit 先は main)
- predict_core / daily_predict / V15 model: 不変 verification 済

## test 1: kill-switch
```
$ touch data/v18/save_all_horse_scores.kill
$ python tools/save_all_horse_scores.py --date 20260510
[save_all_horse_scores] kill-switch active (...) → no-op exit
```
✓ 即 no-op exit、 model load にも到達せず。

## test 2: csv 不在 handling
```
$ rm data/v18/save_all_horse_scores.kill
$ python tools/save_all_horse_scores.py --date 20260510
[STEP 0] V15 model load... OK (is_live=True)
[INFO] 20260510 の race 一覧見つからず (daily_predict.py 未完了 or 非開催)
```
✓ csv 不在で clean exit、 sys.exit せず正常 return。

## test 3: import sanity
```
$ python -c "from predict_core import load_models, parse_shutuba, predict_race; from jrdb_features import merge_jrdb_predict_features; print('imports OK')"
imports OK
```
✓ 全関数 import OK。

## test 4: 単一 R inference (過去 R、 想定内 fail)
```
$ python tools/save_all_horse_scores.py --date 20260509 --race-id 202604010312 --dry-run
[1/1] race_id=202604010312
  [SKIP] 馬データなし
[STEP 2] inference 完了 (0.4s, 1 R, 0 row, 1 fail)
[WARN] 0 row、 csv 保存スキップ
```
- 過去 R (5/9) の shutuba page は期限切れで `parse_shutuba` が 0 馬を返す
- これは LEAK 防止上 望ましい挙動 (過去 R に対する retroactive inference を意図的に避ける)
- 5/10 朝 9:30 schtask fire 時は **未来 R** (5/10 当日) なので shutuba page 有効、 正常動作見込み

## test 5: 既存 file 不変 verification
```
$ git status -s tools/predict_core.py tools/daily_predict.py app.py
(空)

$ md5sum keiba_model_v15_central*.pkl.gz
309dffc65504f056d233c65665c319d5 *keiba_model_v15_central.pkl.gz
fac1588ac20e96ae81eef1efbf7f423e *keiba_model_v15_central_live.pkl.gz
```
✓ predict_core / daily_predict / app.py / V15 model 全 不変。

## test 6: 5/10 朝 fire 計画 (manual fallback)

schtask 9:30 fire が万一動かない場合の手動コマンド:
```
$ python tools/save_all_horse_scores.py --date 20260510
```
or
```
$ wscript.exe tools/silent_runner.vbs save_all_horse_scores_runner.bat --date 20260510
```

## smoke test 残課題

5/10 朝 9:30+ で実 fire した後、 `data/daily_predictions_full/20260510.csv` に
- row 数 (期待: 30-40 R × 8-18 頭 = 240-720 row)
- V15_score 値 0-1 範囲
- 全 36 R cover
- source 列 = "v15_production_full"

を確認。 Session #72 (5/10 後) の audit task 候補。
