# Session #70 A: 5月 production saved score audit

作成: 2026-05-09 17:15 JST
branch: dev/audit-backtest

## ★ LEAK PREVENTION 宣言 ★

本 audit 〜 Session #70 全領域 で **以下を一切使用しない**:

- ❌ V15 model.predict() / pkl/joblib model load
- ❌ tools/predict_core.py / tools/daily_predict.py / app.py 実行
- ❌ feature engineering を伴う任意の inference
- ❌ `data/v18/session69_horse_scores.csv` (Session #69 で再 inference 済 = LEAK 判定)

すべて **production 時点 (各日朝 8:00 schtask) で predict + 保存済** の score のみを read-only で参照する。

## 1. 利用可能な production saved source (5月分)

### 1-1. data/daily_predictions/{YYYYMMDD}.csv

```
$ ls data/daily_predictions/2026050*.csv
data/daily_predictions/20260509.csv  (7,577 bytes, mtime 2026-05-09 08:56)
```

**5/9 のみ** 存在。 5/1-5/8 は不在 (5/2, 5/3 の本番予測は別 source = `data/cumulative_results.csv` に集約されている)。

#### 20260509.csv columns
race_id, course, race_num, race_name, condition, num_horses, distance, surface, track_condition,
top1_num, top1_name, top1_score, top2_num, top2_name, top3_num, top3_name, trio_bets, bet_type, investment

→ 各馬 score は top1/top2/top3 のみ保存 (4 着以下の score なし)。

### 1-2. data/cumulative_results.csv

production 済 (実結果 + 払戻 join 済) の累積記録。 5月分:

| date | row 数 | 11R | 12R |
|---|---|---|---|
| 20260502 | 12 | 3 | 2 |
| 20260503 | 12 | 3 | 3 |
| 計 | 24 | 6 | 5 |

#### columns (関連)
race_id, course, race_num, race_name, condition, num_horses, distance, surface, track_condition,
top1_num, top1_name, top1_score, top2_num, top3_num,
top1_finish, top2_finish, top3_finish,
trio_bets, trio_result, bet_type, trio_hit, trio_payout, umaren_payout, actual_payout,
investment, profit, status, date, umaren_hit, trio_bets_str

→ 5/2, 5/3 の 11R + 12R 11 records は ★ production saved ★ (V15 prediction + 実結果 + 払戻、 全て当日処理済)。

## 2. 5月分 集計 source 一覧

| source | 期間 | 11R+12R 候補数 | 重賞除外後 |
|---|---|---|---|
| data/cumulative_results.csv | 5/2-5/3 | 11 | filter で確定 (B) |
| data/daily_predictions/20260509.csv | 5/9 | 6 | filter で確定 (B) |
| **計** | 5/2-5/3, 5/9 | **17** | (B 出力参照) |

5/1, 5/4-5/8 は production saved データなし (= 開催なし or 当該 csv 未保存)。

## 3. 不在日の扱い

5/4 (日) は中央開催あったが daily_predictions 不在 → cumulative_results にも未収録 → ★本 session では対象外★。 後続 session で別 source (例: data/track_record.csv) から復元するか検討。

## 4. ★ session69_horse_scores.csv は使わない ★

`data/v18/session69_horse_scores.csv` は Session #69 で過去 R に対し再 inference した産物 → V15 学習 cutoff 後の特徴量を含む可能性あり = **LEAK 判定**。 本 session では一切参照しない。
