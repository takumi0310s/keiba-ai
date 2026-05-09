# Session #61 A: 入力 status (5/9)

## Session #60 出力 確認

| ファイル | 状態 | 内容 |
|---|---|---|
| data/v18/horse_motion_5_9.csv | OK | 9 行 (3 race × 3 頭) simulate 値 のみ |
| data/v18/predictions_majors_5system_5_9_FINAL.json | OK | 5 system × 3 race、 V15 top3 + system 5 simulate |
| data/v18/predictions_5_9_all.json | OK | 36 race、 京都/新潟 11R は v15_scores 全 16 頭、 東京 11R のみ scores 0 件 |

## 重賞 3R race_id

| race_id | 場 | R | レース名 | 発走 | 全馬 v15_scores |
|---|---|---|---|---|---|
| 202605020511 | 東京 | 11 | エプソムカップ (G3) | 15:45 | 0 件 (top3 のみ FINAL.json) |
| 202608030511 | 京都 | 11 | 京都新聞杯 (G2) | 15:30 | 16 頭 |
| 202604010311 | 新潟 | 11 | 駿風 S (OP) | 15:20 | 16 頭 |

## 動画 motion 状態

simulate 値のみ (Session #60 B で 動画 DL 失敗 HTTP 400)。
各 race 3 頭 (sim_X_1/2/3) しか motion なし。
=> 全馬 score = V15 score ベース + motion 3 頭は補助情報。

## 評価方針

- integrated_score = V15 score の race 内 percentile (0-1)
- 動画 motion 3 頭ある horse_id は "sim 代表" として補助表示
- confidence: video=high (動画あり), static=low (V15 のみ)
- 東京 11R は top3 のみ表示 (fallback)

決定: Session #60 完了確認済 → B (scoring 実装) 進行。
