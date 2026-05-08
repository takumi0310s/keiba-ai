# Session #47 C: 5/9 全 R 予測 pre-compute (2026-05-08)

## 1. 目的

5/9 (土) 中央 全 R (~36 R) を V15 で予測 + 拡張調教 features 効果を並列計算。
5/10 朝の verdict (Session #47 D) 用 baseline。

## 2. tool

`tools/predict_all_5_9.py`

```bash
python tools/predict_all_5_9.py                 # 5/9 全 R
python tools/predict_all_5_9.py --date 20260509 # 日付指定
python tools/predict_all_5_9.py --limit 3       # 3 R smoke test
python tools/predict_all_5_9.py --no-extended   # V15 only
```

## 3. 入出力

### 入力
- 5/9 出馬表 (netkeiba、 fetch_race_list 経由)
- V15 model (`keiba_model_v15_central_live.pkl.gz`、 md5 verify)

### 出力
- `data/v18/predictions_5_9_all.json` — 全 R 予測

format:
```json
{
  "date": "20260509",
  "v15_md5": "309dffc6...",
  "race_count": 36,
  "predictions": [
    {
      "race_id": "202608030611",
      "venue": "kyoto",
      "race_num": 11,
      "race_name": "京都新聞杯",
      "grade": "G2",
      "num_horses": 16,
      "v15_top3": [...],
      "v15_scores": {...},
      "extended_top3": {...},
      "error": null
    },
    ...
  ]
}
```

## 4. V15 model md5 (重要)

CLAUDE.md 記載値: `842b9a5f305c793ed8fa54a74e06b836` (古い)
**現在の実 md5**: `309dffc65504f056d233c65665c319d5`

→ 実 md5 を baseline とし、 不変保証 する。
→ Session #47 期間中 model file 一切変更しない (保証)。

## 5. 5/9 race 構成 (検出済 36 R)

```
[19:29:14] Fetched 36 races for 20260509
```

3 場 × 12 R = 36 R (京都 / 東京 / 新潟)

### 重賞 3 R (予測 + 観戦のみ、 投票しない)
- 東京 11R エプソムC (G3) 15:45
- 京都 11R 京都新聞杯 (G2) 15:30
- 新潟 11R 駿風 S (OP) 15:20

### 投資 R (V15 案B改)
- 12R 1勝クラス のみ、 各場 最大 1 R、 700 円 × 最大 3 R = 2,100 円 上限

## 6. 拡張調教 features (Session #47 B)

V15 (150 features) と V15 + 拡張調教 (158 features) で並列予測予定。
拡張 8 features:
- training_time_5f / training_time_3f / training_pace_5f_3f
- days_since_last_training / training_count_2w
- cyb_train_baba_enc / cyb_train_amount / cyb_train_change_enc

⚠️ **注**: 現状 C は **V15 baseline のみ予測**。
拡張版予測は B AUC test 結果 (採用基準達成) 確認後に
追加実装 + 再 run 想定 (V20 候補としての評価)。

## 7. 実行 log (5/8 19:29 開始)

```
[19:29:13] === Session #47 C: predict 20260509 all races ===
[19:29:13] V15 model md5: 309dffc65504f056d233c65665c319d5
[19:29:13]   OK: V15 model 不変
[19:29:14] Fetched 36 races for 20260509
[19:29:14] [1/36] 202608030501 京都1R
...
```

ETA: ~ 20 分 (1 R ~ 30-40 秒)。
完了後 `data/v18/predictions_5_9_all.json` 確認。

## 8. リーク監査

- 予測は **5/8 夕方 (レース前)** 実行
- 当日情報 (オッズ / 馬体重 / 馬場) は **5/9 朝発表** で
  予測時点では取得不可 → defaults / 過去値 で代替
- 結果との照合は Session #47 D (5/10 朝)
- timestamp + md5 を JSON に記録 → 再現性保証

## 9. 5/9 朝の運用 (V15 案B改 維持、 絶対)

C の予測は 5/10 verdict 用 **学習目的のみ**。
5/9 朝の **本番投票** は **既存 V15 daily_predict.py が担当**:

```
06:00  daily_jrdb_kyi (JRDB 取得)
07:30  jrdb_health_check
08:00  daily_predict (V15 全 R 予測 → CSV)
08:45  race_auto_notify (5 分前 Discord 投票推奨)
```

C は production flow に **影響しない** (read-only、 別 JSON)。

## 10. 採用判定 (5/10 朝 D verdict 後)

| 結果 | judgment |
|------|----------|
| top1 hit ≥ 30%、 grade-monotonic | V15 baseline 健全 |
| top1 hit < 25% | 5/9 サンプル不足、 5/16, 5/17 で再計測 |
| クラス別 outlier | 該当 grade で V15 弱点 確認 → V20 課題化 |

## 11. 関連 file

- `tools/predict_all_5_9.py` (本 tool)
- `tools/daily_predict.py` (production、 read-only 流用 fetch_race_list)
- `tools/predict_one_race.py` (1R 予測 流用)
- `keiba_model_v15_central_live.pkl.gz` (V15 model)
- `data/v18/predictions_5_9_all.json` (出力)
