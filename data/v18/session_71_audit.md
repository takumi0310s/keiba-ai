# Session #71 A: 既存 logic audit (5/9 18:01)

## 1. 全頭 score 計算箇所

`tools/predict_core.py` L2148 `predict_race(df, model_data, odds_available, race_info)`:
- 入力 df は **全出走馬** (parse_shutuba で取得した全 `horses` を build_features に通したもの)
- LGB `predict_proba` / `predict` で `ai_scores` を全行に対して計算
- XGB / CatBoost / FT-Transformer ensemble (重み付き和) で最終 `df['スコア']` を全頭に書き込む
- ★ 戻り値 df は全頭分の score を持つ ★

## 2. top3 切り出し箇所

`tools/daily_predict.py` L381:
```
sorted_df = df.sort_values('スコア', ascending=False).reset_index(drop=True)
...
top1 = sorted_df.iloc[0]
top2 = sorted_df.iloc[1]
top3 = sorted_df.iloc[2]
```

L403-422 で row dict 組み立て時に **top1/2/3 のみ抽出** → CSV 1 行/R で書く (4 着以下の score は捨てられる)。

## 3. race loop 構造

`tools/daily_predict.py` L233 `run_daily_predict(date_str, ...)`:
1. `load_models()` (model 1 回 load)
2. `fetch_race_list(date_str)` (race_id list 取得)
3. for race in races: parse_shutuba → odds → JRA天候 → get_horse_stats × N → build_features → merge_jrdb_predict_features → predict_race → top3 cut → `_append_prediction_to_csv`

## 4. Session #71 設計

`predict_race` は全頭 score を返してくれるので、 daily_predict.py の race loop と同じ flow を **新 file** で reproduce し、 top3 cut 直前で `sorted_df` を全行 CSV に書き出す。

★ 既存 logic は一切 触らない ★。 predict_core / daily_predict / app.py / V15 model file 全て read-only import のみ。

## 5. 並行実行 安全性

- daily_predict.py が 8:00-8:56 で daily_predictions/{date}.csv を append しながら書く
- save_all_horse_scores.py は schtask 9:30 fire (daily_predict 完了後) で daily_predictions/ csv を **read-only** で race_id list 取得 → 別 directory `daily_predictions_full/` に write
- 衝突なし (read source / write target 完全分離)

## 6. ★ NEVER 確認 ★

- predict_core.py 修正 → ✗ なし (read-only import)
- daily_predict.py 修正 → ✗ なし
- app.py 修正 → ✗ なし
- V15 model 修正 → ✗ なし (md5 fac1588ac... / 309dffc65... 確認、 不変)
- 既存 schtasks 49 件 → ✗ 不変、 +1 件のみ
- daily_predict.py / race_auto_notify.py を trigger → ✗ なし (subprocess 不使用、 純 import)
